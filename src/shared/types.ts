
export type TensorId = string;

// --- Main Thread <-> Coordinator ---

export type CoordinatorRequest = 
    | { type: 'INIT_COORDINATOR', id: string, payload: { buffer: SharedArrayBuffer, totalWorkers: number } }
    | { type: 'ADD_WORKER', payload: { workerId: number } } // Transfer port in second arg
    | { type: 'ALLOC', payload: { id: TensorId, size: number } }
    | { type: 'ALLOC_VIEW', payload: { id: TensorId, parentId: TensorId, offset?: number } } // Create view that shares parent's memory
    | { type: 'FREE', payload: { id: TensorId } }
    | { type: 'SET', payload: { id: TensorId, offset: number, value: number } }
    | { type: 'WRITE', payload: { id: TensorId, data: Float32Array } }
    | { type: 'OP', id?: string, payload: { op: string, inputs: TensorId[], output: TensorId, params?: any } }
    | { type: 'READ', id: string, payload: { id: TensorId } }
    | { type: 'READ_VIEW', id: string, payload: { id: TensorId } }
    | { type: 'READ_VALUE', id: string, payload: { id: TensorId, offset: number } };

export type CoordinatorResponse = 
    | { id: string, data: any, error?: string };

// --- Coordinator <-> Compute Worker ---

export type ComputeRequest = 
    | { type: 'INIT_WORKER', payload: { workerId: number, role: 'COMPUTE', buffer: SharedArrayBuffer } } // Transfer port in second arg
    | { 
        type: 'EXECUTE_TASK', 
        taskId: string, 
        op: string, 
        inputs: { offset: number, size: number }[], 
        output: { offset: number, size: number }, 
        params: any, 
        workerIndex: number, 
        totalWorkers: number 
      };

export type ComputeResponse = 
    | { type: 'TASK_DONE', taskId: string };

// --- Typed Wrapper ---

export class TypedWorker<Req, Res> {
    private worker: Worker;

    constructor(worker: Worker) {
        this.worker = worker;
    }

    postMessage(message: Req, transfer?: Transferable[]) {
        this.worker.postMessage(message, transfer || []);
    }

    onMessage(handler: (data: Res) => void) {
        this.worker.onmessage = (event) => handler(event.data);
    }
    
    onError(handler: (err: ErrorEvent) => void) {
        this.worker.onerror = handler;
    }

    terminate() {
        this.worker.terminate();
    }
    
    // Expose raw worker if needed (e.g. for addEventListener)
    get raw(): Worker {
        return this.worker;
    }
}

export class TypedPort<Req, Res> {
    private port: MessagePort;

    constructor(port: MessagePort) {
        this.port = port;
    }

    postMessage(message: Req, transfer?: Transferable[]) {
        this.port.postMessage(message, transfer || []);
    }

    onMessage(handler: (data: Res) => void) {
        this.port.onmessage = (event) => handler(event.data);
    }
}
