import { TypedWorker, CoordinatorRequest, CoordinatorResponse, ComputeRequest } from '../types';

export class Dispatcher {
    private coordinator: TypedWorker<CoordinatorRequest, CoordinatorResponse> | null = null;
    private computeWorkers: Worker[] = []; // We don't talk to these directly much
    private callbacks: Map<string, (data: any) => void> = new Map();
    private tensorIdCounter: number = 0;

    // Singleton pattern
    private static _instance: Dispatcher;

    private constructor() {}

    public static get instance(): Dispatcher {
        if (!Dispatcher._instance) {
            Dispatcher._instance = new Dispatcher();
        }
        return Dispatcher._instance;
    }

    async init(workerScriptUrl: string = './worker.js', threadCount: number = 4, memorySizeMB: number = 256): Promise<void> {
        if (this.coordinator) return; // Already initialized

        // Create SharedArrayBuffer
        const sab = new SharedArrayBuffer(1024 * 1024 * memorySizeMB);

        // 1. Spawn Coordinator
        const coordWorker = new Worker(new URL(workerScriptUrl, import.meta.url), { type: 'module' });
        this.coordinator = new TypedWorker(coordWorker);
        this.setupWorkerHandler(this.coordinator, 'Coordinator');

        // 2. Spawn Compute Workers and establish channels
        for (let i = 0; i < threadCount; i++) {
            const worker = new Worker(new URL(workerScriptUrl, import.meta.url), { type: 'module' });
            this.computeWorkers.push(worker);
            
            // We don't necessarily need to listen to compute workers in the main thread
            // unless we want to catch errors.
            worker.onerror = (err) => console.error(`Compute-${i} System Error:`, err);

            // Create a direct channel between Coordinator and this Compute Worker
            const channel = new MessageChannel();

            // Send port1 to Coordinator
            this.coordinator.postMessage({
                type: 'ADD_WORKER',
                payload: { workerId: i }
            }, [channel.port1]); // Transfer ownership

            // Send port2 to Compute Worker
            // We manually post here because ComputeRequest is for the internal channel, 
            // but INIT_WORKER is special as it comes from Main
            const initMsg: ComputeRequest = {
                type: 'INIT_WORKER',
                payload: { 
                    workerId: i,
                    role: 'COMPUTE',
                    buffer: sab
                }
            };
            worker.postMessage(initMsg, [channel.port2]);
        }

        // 3. Initialize Coordinator with SAB
        return new Promise((resolve) => {
            const reqId = this.generateId();
            this.callbacks.set(reqId, () => resolve());
            
            this.coordinator!.postMessage({
                type: 'INIT_COORDINATOR',
                id: reqId,
                payload: { 
                    buffer: sab,
                    totalWorkers: threadCount
                }
            });
        });
    }

    private setupWorkerHandler(worker: TypedWorker<any, CoordinatorResponse>, name: string) {
        worker.onMessage((data) => {
            const { id, data: responseData, error } = data;
            
            if (id && this.callbacks.has(id)) {
                const callback = this.callbacks.get(id)!;
                if (error) {
                    console.error(`${name} Error:`, error);
                } else {
                    callback(responseData);
                }
                this.callbacks.delete(id);
            } else if (error) {
                console.error(`${name} Unhandled Error:`, error);
            }
        });
        
        worker.onError((err) => {
            console.error(`${name} System Error:`, err);
        });
    }

    // Generate a unique ID for a new tensor
    nextTensorId(): string {
        return `t_${this.tensorIdCounter++}`;
    }

    // --- Commands (Sent to Coordinator) ---

    allocate(tensorId: string, size: number): void {
        this.postToCoordinator({
            type: 'ALLOC',
            payload: { id: tensorId, size }
        });
    }

    free(tensorId: string): void {
        this.postToCoordinator({
            type: 'FREE',
            payload: { id: tensorId }
        });
    }

    runOp(op: string, inputs: string[], output: string, params: any = {}): void {
        this.postToCoordinator({
            type: 'OP',
            payload: { op, inputs, output, params }
        });
    }

    read(tensorId: string): Promise<Float32Array> {
        return new Promise((resolve) => {
            const reqId = this.generateId();
            this.callbacks.set(reqId, (data) => {
                resolve(data.data); 
            });

            this.postToCoordinator({
                type: 'READ',
                id: reqId,
                payload: { id: tensorId }
            });
        });
    }

    private postToCoordinator(command: CoordinatorRequest) {
        if (!this.coordinator) {
            throw new Error("Dispatcher not initialized. Call init() first.");
        }
        this.coordinator.postMessage(command);
    }

    private generateId(): string {
        return Math.random().toString(36).substring(2, 15);
    }
}
