import { MemoryAllocator } from './memory';
import * as elementwise from '../kernels/elementwise';
import { defineWorkerOnMessage, definePortOnMessage } from '../utils';
import { CoordinatorRequest, ComputeRequest, ComputeResponse, TypedPort } from '../types';

// Types
type WorkerRole = 'COORDINATOR' | 'COMPUTE';

interface TensorMetadata {
    offset: number;
    size: number;
}

// Global State
let role: WorkerRole = 'COORDINATOR'; // Default, changes on init
let memoryAllocator: MemoryAllocator | null = null;
let buffer: SharedArrayBuffer | null = null;
let tensorRegistry: Map<string, TensorMetadata> = new Map();

// Coordinator State
let computePorts: TypedPort<ComputeRequest, ComputeResponse>[] = [];
let pendingTasks: Map<string, { resolve: () => void, count: number }> = new Map();

// Compute State
let coordinatorPort: TypedPort<ComputeResponse, ComputeRequest> | null = null;

// --- Message Handlers ---

// We use a union type for the handler because the worker can receive both types of messages
// depending on its role (initially it receives CoordinatorRequest or ComputeRequest)
// Actually, the main thread sends CoordinatorRequest (mostly) or ComputeRequest (INIT_WORKER)
// Let's just cast inside for simplicity or use a union type if we want to be strict.
// Since defineWorkerOnMessage takes a generic T, we can use CoordinatorRequest | ComputeRequest

self.onmessage = defineWorkerOnMessage<CoordinatorRequest | ComputeRequest>((data, ports) => {
    const { type } = data;

    if (type === 'INIT_COORDINATOR') {
        role = 'COORDINATOR';
        const payload = (data as any).payload; // TS might struggle with the union discrimination here without a switch
        buffer = payload.buffer;
        memoryAllocator = new MemoryAllocator(buffer!);
        // Acknowledge init
        self.postMessage({ id: (data as any).id, data: { status: 'ok' } });
        return;
    }

    if (type === 'INIT_WORKER') {
        role = 'COMPUTE';
        const payload = (data as any).payload;
        buffer = payload.buffer;
        
        // The port comes in the transfer list
        const port = ports[0];
        coordinatorPort = new TypedPort(port);
        setupComputeWorker(coordinatorPort);
        return;
    }

    if (type === 'ADD_WORKER') {
        if (role !== 'COORDINATOR') return;
        const port = ports[0];
        const typedPort = new TypedPort<ComputeRequest, ComputeResponse>(port);
        computePorts.push(typedPort);
        setupCoordinatorPort(typedPort);
        return;
    }

    // Coordinator Commands
    if (role === 'COORDINATOR') {
        const req = data as CoordinatorRequest;
        switch (req.type) {
            case 'ALLOC':
                handleAlloc(req.payload);
                break;
            case 'FREE':
                handleFree(req.payload);
                break;
            case 'OP':
                handleOp({ ...req.payload, params: req.payload.params || {} }, req.id);
                break;
            case 'READ':
                handleRead(req.payload, req.id);
                break;
        }
    }
});

// --- Coordinator Logic ---

function setupCoordinatorPort(port: TypedPort<ComputeRequest, ComputeResponse>) {
    port.onMessage((data) => {
        if (data.type === 'TASK_DONE') {
            const task = pendingTasks.get(data.taskId);
            if (task) {
                task.count--;
                if (task.count === 0) {
                    task.resolve();
                    pendingTasks.delete(data.taskId);
                }
            }
        }
    });
}

function handleAlloc(payload: { id: string, size: number }) {
    if (!memoryAllocator) return;
    try {
        const offset = memoryAllocator.allocate(payload.size);
        tensorRegistry.set(payload.id, { offset, size: payload.size });
    } catch (e: any) {
        console.error("Allocation failed:", e.message);
    }
}

function handleFree(payload: { id: string }) {
    if (!memoryAllocator) return;
    const meta = tensorRegistry.get(payload.id);
    if (meta) {
        memoryAllocator.free(meta.offset, meta.size);
        tensorRegistry.delete(payload.id);
    }
}

async function handleOp(payload: { op: string, inputs: string[], output: string, params: any }, reqId?: string) {
    // 1. Resolve Tensor IDs to Offsets
    const inputMetas = payload.inputs.map(id => tensorRegistry.get(id));
    const outputMeta = tensorRegistry.get(payload.output);

    if (inputMetas.some(m => !m) || !outputMeta) {
        console.error("Missing tensor metadata for op:", payload.op);
        return;
    }

    // 2. Split the work (Sharding Logic)
    const numWorkers = computePorts.length;
    const taskId = Math.random().toString(36).substring(7);

    const donePromise = new Promise<void>((resolve) => {
        pendingTasks.set(taskId, { resolve, count: numWorkers });
    });

    // 3. Dispatch to Compute Workers
    computePorts.forEach((port, index) => {
        port.postMessage({
            type: 'EXECUTE_TASK',
            taskId,
            op: payload.op,
            inputs: inputMetas.map(m => ({ offset: m!.offset, size: m!.size })),
            output: { offset: outputMeta!.offset, size: outputMeta!.size },
            params: payload.params,
            workerIndex: index,
            totalWorkers: numWorkers
        });
    });

    // 4. Wait and Reply
    await donePromise;
    
    if (reqId) {
        self.postMessage({ id: reqId, data: { status: 'done' } });
    }
}

function handleRead(payload: { id: string }, reqId: string) {
    const meta = tensorRegistry.get(payload.id);
    if (!meta || !buffer) return;

    const src = new Float32Array(buffer, meta.offset, meta.size / 4);
    const copy = new Float32Array(src);

    self.postMessage({ 
        id: reqId, 
        data: { data: copy } 
    }, [copy.buffer]);
}

// --- Compute Logic ---

function setupComputeWorker(port: TypedPort<ComputeResponse, ComputeRequest>) {
    port.onMessage((data) => {
        if (data.type === 'EXECUTE_TASK') {
            executeKernel(data.op, data.inputs, data.output, data.params, data.workerIndex, data.totalWorkers);
            port.postMessage({ type: 'TASK_DONE', taskId: data.taskId });
        }
    });
}

function executeKernel(op: string, inputs: any[], output: any, params: any, workerIndex: number, totalWorkers: number) {
    if (!buffer) return;

    const inputViews = inputs.map(meta => new Float32Array(buffer!, meta.offset, meta.size / 4));
    const outputView = new Float32Array(buffer!, output.offset, output.size / 4);

    const totalElements = outputView.length;
    const chunkSize = Math.ceil(totalElements / totalWorkers);
    const start = workerIndex * chunkSize;
    const end = Math.min(start + chunkSize, totalElements);

    if (start >= totalElements) return;

    switch (op) {
        case 'ADD':
            elementwise.add(inputViews[0], inputViews[1], outputView, start, end);
            break;
        case 'SUB':
            elementwise.sub(inputViews[0], inputViews[1], outputView, start, end);
            break;
        case 'MUL':
            elementwise.mul(inputViews[0], inputViews[1], outputView, start, end);
            break;
        case 'DIV':
            elementwise.div(inputViews[0], inputViews[1], outputView, start, end);
            break;
        case 'RELU':
            elementwise.relu(inputViews[0], outputView, start, end);
            break;
        case 'EXP':
            elementwise.exp(inputViews[0], outputView, start, end);
            break;
        case 'LOG':
            elementwise.log(inputViews[0], outputView, start, end);
            break;
        case 'FILL':
            elementwise.fill(outputView, params.value, start, end);
            break;
        case 'RANDN':
            elementwise.randn(outputView, start, end);
            break;
        default:
            console.error(`Unknown op: ${op}`);
    }
}
