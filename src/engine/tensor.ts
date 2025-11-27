import { Dispatcher } from '../backend/dispatcher';

// Automatic memory management
const registry = new FinalizationRegistry((id: string) => {
    Dispatcher.instance.free(id);
});

export class Tensor {
    id: string;
    shape: number[];
    requiresGrad: boolean;
    grad: Tensor | null = null;
    
    // Graph for Autograd
    op: string | null = null;
    prev: Tensor[] = [];

    constructor(id: string, shape: number[], requiresGrad: boolean = false) {
        this.id = id;
        this.shape = shape;
        this.requiresGrad = requiresGrad;

        // Register for automatic cleanup when this JS object is GC'd
        registry.register(this, this.id);
    }

    // --- Initialization ---

    static async init(threads: number = 4) {
        // We don't pass a path here, letting Dispatcher use its default relative path to worker.ts
        await Dispatcher.instance.init(undefined, threads);
    }

    static zeros(shape: number[], requiresGrad: boolean = false): Tensor {
        return Tensor.create(shape, requiresGrad, 'FILL', { value: 0 });
    }

    static randn(shape: number[], requiresGrad: boolean = false): Tensor {
        return Tensor.create(shape, requiresGrad, 'RANDN');
    }

    private static create(shape: number[], requiresGrad: boolean, op: string, params: any = {}): Tensor {
        const size = shape.reduce((a, b) => a * b, 1) * 4; // 4 bytes per float
        const id = Dispatcher.instance.nextTensorId();
        
        Dispatcher.instance.allocate(id, size);
        Dispatcher.instance.runOp(op, [], id, params);

        return new Tensor(id, shape, requiresGrad);
    }

    // --- Operations ---

    add(other: Tensor): Tensor {
        return this.runBinaryOp('ADD', other);
    }

    sub(other: Tensor): Tensor {
        return this.runBinaryOp('SUB', other);
    }

    mul(other: Tensor): Tensor {
        return this.runBinaryOp('MUL', other);
    }

    div(other: Tensor): Tensor {
        return this.runBinaryOp('DIV', other);
    }

    relu(): Tensor {
        return this.runUnaryOp('RELU');
    }

    exp(): Tensor {
        return this.runUnaryOp('EXP');
    }

    log(): Tensor {
        return this.runUnaryOp('LOG');
    }

    // --- Data Access ---

    async toArray(): Promise<Float32Array> {
        return await Dispatcher.instance.read(this.id);
    }

    async item(): Promise<number> {
        const arr = await this.toArray();
        return arr[0];
    }

    // --- Autograd ---

    backward() {
        if (!this.requiresGrad) return;
        // TODO: Implement topological sort and backward pass
        console.warn("Backward not implemented yet");
    }

    // --- Internal Helpers ---

    private runBinaryOp(op: string, other: Tensor): Tensor {
        if (!this.shapeEquals(other.shape)) {
            throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
        }

        const outId = Dispatcher.instance.nextTensorId();
        const size = this.numElements() * 4;
        
        Dispatcher.instance.allocate(outId, size);
        Dispatcher.instance.runOp(op, [this.id, other.id], outId);

        const out = new Tensor(outId, this.shape, this.requiresGrad || other.requiresGrad);
        out.op = op;
        out.prev = [this, other];
        
        return out;
    }

    private runUnaryOp(op: string): Tensor {
        const outId = Dispatcher.instance.nextTensorId();
        const size = this.numElements() * 4;
        
        Dispatcher.instance.allocate(outId, size);
        Dispatcher.instance.runOp(op, [this.id], outId);

        const out = new Tensor(outId, this.shape, this.requiresGrad);
        out.op = op;
        out.prev = [this];

        return out;
    }

    private shapeEquals(other: number[]): boolean {
        if (this.shape.length !== other.length) return false;
        for (let i = 0; i < this.shape.length; i++) {
            if (this.shape[i] !== other[i]) return false;
        }
        return true;
    }

    private numElements(): number {
        return this.shape.reduce((a, b) => a * b, 1);
    }
}
