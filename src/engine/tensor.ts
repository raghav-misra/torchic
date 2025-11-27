import { Dispatcher } from '../backend/dispatcher';

// Automatic memory management
const registry = new FinalizationRegistry((id: string) => {
    Dispatcher.instance.free(id);
});

export class Tensor {
    id: string;
    shape: number[];
    strides: number[];
    requiresGrad: boolean;
    grad: Tensor | null = null;
    
    // Graph for Autograd
    op: string | null = null;
    prev: Tensor[] = [];

    constructor(id: string, shape: number[], requiresGrad: boolean = false) {
        this.id = id;
        this.shape = shape;
        this.strides = Tensor.computeStrides(shape);
        this.requiresGrad = requiresGrad;

        // Register for automatic cleanup when this JS object is GC'd
        registry.register(this, this.id);
    }

    private static computeStrides(shape: number[]): number[] {
        const strides = new Array(shape.length);
        let stride = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
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

    static fromData(data: Float32Array | number[], shape: number[], requiresGrad: boolean = false): Tensor {
        const size = shape.reduce((a, b) => a * b, 1) * 4;
        const id = Dispatcher.instance.nextTensorId();
        
        Dispatcher.instance.allocate(id, size);
        
        // Convert to Float32Array if needed
        const typedData = data instanceof Float32Array ? data : new Float32Array(data);
        Dispatcher.instance.write(id, typedData);

        return new Tensor(id, shape, requiresGrad);
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

    setValue(indices: number[], value: number) {
        // Calculate flat index
        let flatIndex = 0;
        let stride = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            flatIndex += indices[i] * stride;
            stride *= this.shape[i];
        }
        
        Dispatcher.instance.set(this.id, flatIndex, value);
   }

    async getValue(indices: number[]): Promise<number> {
        // Calculate flat index
        let flatIndex = 0;
        let stride = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            flatIndex += indices[i] * stride;
            stride *= this.shape[i];
        }
        
        return await Dispatcher.instance.readValue(this.id, flatIndex);
    }

    matmul(other: Tensor): Tensor {
        if (this.shape.length !== 2 || other.shape.length !== 2) {
            throw new Error(`MatMul requires 2D tensors. Got ${this.shape} and ${other.shape}`);
        }
        if (this.shape[1] !== other.shape[0]) {
            throw new Error(`Shape mismatch for MatMul: ${this.shape} vs ${other.shape}`);
        }

        const m = this.shape[0];
        const k = this.shape[1];
        const n = other.shape[1];
        const outShape = [m, n];

        const outId = Dispatcher.instance.nextTensorId();
        const size = m * n * 4;
        
        Dispatcher.instance.allocate(outId, size);
        Dispatcher.instance.runOp('MATMUL', [this.id, other.id], outId, { m, n, k });

        const out = new Tensor(outId, outShape, this.requiresGrad || other.requiresGrad);
        out.op = 'MATMUL';
        out.prev = [this, other];
        
        return out;
    }

    transpose(): Tensor {
        if (this.shape.length !== 2) {
            throw new Error("Transpose only supported for 2D tensors");
        }
        const [m, n] = this.shape;
        const outShape = [n, m];
        
        const outId = Dispatcher.instance.nextTensorId();
        const size = m * n * 4;
        
        Dispatcher.instance.allocate(outId, size);
        Dispatcher.instance.runOp('TRANSPOSE', [this.id], outId, { m, n });

        const out = new Tensor(outId, outShape, this.requiresGrad);
        out.op = 'TRANSPOSE';
        out.prev = [this];
        
        return out;
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

        // 1. Topological Sort
        const topo: Tensor[] = [];
        const visited = new Set<string>();
        
        const buildTopo = (v: Tensor) => {
            if (visited.has(v.id)) return;
            visited.add(v.id);
            for (const child of v.prev) {
                buildTopo(child);
            }
            topo.push(v);
        };
        buildTopo(this);

        // 2. Initialize Grad
        this.grad = Tensor.create(this.shape, false, 'FILL', { value: 1.0 });

        // 3. Reverse Pass
        for (let i = topo.length - 1; i >= 0; i--) {
            const v = topo[i];
            if (!v.grad) continue;

            if (v.op === 'ADD') {
                const [a, b] = v.prev;
                if (a.requiresGrad) a.addGrad(v.grad);
                if (b.requiresGrad) b.addGrad(v.grad);
            } else if (v.op === 'SUB') {
                const [a, b] = v.prev;
                if (a.requiresGrad) a.addGrad(v.grad);
                if (b.requiresGrad) b.addGrad(v.grad.neg());
            } else if (v.op === 'MUL') {
                const [a, b] = v.prev;
                if (a.requiresGrad) a.addGrad(v.grad.mul(b));
                if (b.requiresGrad) b.addGrad(v.grad.mul(a));
            } else if (v.op === 'DIV') {
                const [a, b] = v.prev;
                if (a.requiresGrad) a.addGrad(v.grad.div(b));
                if (b.requiresGrad) b.addGrad(v.grad.mul(a).div(b.mul(b)).neg());
            } else if (v.op === 'RELU') {
                const [a] = v.prev;
                if (a.requiresGrad) {
                    // RELU_BACKWARD: gradInput = (input > 0) ? gradOutput : 0
                    const outId = Dispatcher.instance.nextTensorId();
                    Dispatcher.instance.allocate(outId, a.numElements() * 4);
                    Dispatcher.instance.runOp('RELU_BACKWARD', [a.id, v.grad.id], outId);
                    const g = new Tensor(outId, a.shape, false);
                    a.addGrad(g);
                }
            } else if (v.op === 'EXP') {
                const [a] = v.prev;
                if (a.requiresGrad) a.addGrad(v.grad.mul(v));
            } else if (v.op === 'LOG') {
                const [a] = v.prev;
                if (a.requiresGrad) a.addGrad(v.grad.div(a));
            } else if (v.op === 'MATMUL') {
                const [a, b] = v.prev;
                if (a.requiresGrad) a.addGrad(v.grad.matmul(b.transpose()));
                if (b.requiresGrad) b.addGrad(a.transpose().matmul(v.grad));
            } else if (v.op === 'TRANSPOSE') {
                const [a] = v.prev;
                if (a.requiresGrad) a.addGrad(v.grad.transpose());
            } else if (v.op === 'SUM') {
                const [a] = v.prev;
                if (a.requiresGrad) {
                    // Gradient of sum is 1s expanded to input shape
                    // But since we accumulate gradients, we just add the scalar grad to all elements
                    // We handle this in addGrad via broadcasting
                    a.addGrad(v.grad);
                }
            }
        }
    }

    private addGrad(g: Tensor) {
        if (!this.grad) {
            this.grad = g;
        } else {
            // If shapes match, simple add
            if (this.shapeEquals(g.shape)) {
                this.grad = this.grad.add(g);
            } else if (g.numElements() === 1) {
                // Broadcast add (scalar tensor to this tensor)
                // We need a special op for this: ADD_SCALAR_TENSOR
                const outId = Dispatcher.instance.nextTensorId();
                const size = this.numElements() * 4;
                Dispatcher.instance.allocate(outId, size);
                Dispatcher.instance.runOp('ADD_SCALAR_TENSOR', [this.grad.id, g.id], outId);
                this.grad = new Tensor(outId, this.shape, false);
            } else {
                throw new Error(`Gradient shape mismatch: ${this.shape} vs ${g.shape}`);
            }
        }
    }

    neg(): Tensor {
        return this.mul(Tensor.create(this.shape, false, 'FILL', { value: -1 }));
    }

    sum(): Tensor {
        const outId = Dispatcher.instance.nextTensorId();
        const size = 4; // Scalar
        
        Dispatcher.instance.allocate(outId, size);
        Dispatcher.instance.runOp('SUM', [this.id], outId);

        const out = new Tensor(outId, [1], this.requiresGrad);
        out.op = 'SUM';
        out.prev = [this];
        
        return out;
    }

    mean(): Tensor {
        const s = this.sum();
        const n = this.numElements();
        return s.div(Tensor.create([1], false, 'FILL', { value: n }));
    }

    // --- Internal Helpers ---

    private runBinaryOp(op: string, other: Tensor): Tensor {
        const outShape = Tensor.broadcastShapes(this.shape, other.shape);
        const stridesA = Tensor.getBroadcastStrides(this.shape, this.strides, outShape);
        const stridesB = Tensor.getBroadcastStrides(other.shape, other.strides, outShape);

        const outId = Dispatcher.instance.nextTensorId();
        const size = outShape.reduce((a, b) => a * b, 1) * 4;
        
        Dispatcher.instance.allocate(outId, size);
        Dispatcher.instance.runOp(op, [this.id, other.id], outId, { 
            shape: outShape, 
            stridesA, 
            stridesB 
        });

        const out = new Tensor(outId, outShape, this.requiresGrad || other.requiresGrad);
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

    private static broadcastShapes(shapeA: number[], shapeB: number[]): number[] {
        const ndimA = shapeA.length;
        const ndimB = shapeB.length;
        const ndim = Math.max(ndimA, ndimB);
        const outShape = new Array(ndim);
        
        for (let i = 0; i < ndim; i++) {
            const dimA = i < ndim - ndimA ? 1 : shapeA[i - (ndim - ndimA)];
            const dimB = i < ndim - ndimB ? 1 : shapeB[i - (ndim - ndimB)];
            
            if (dimA !== dimB && dimA !== 1 && dimB !== 1) {
                throw new Error(`Shapes ${shapeA} and ${shapeB} are not broadcastable`);
            }
            outShape[i] = Math.max(dimA, dimB);
        }
        return outShape;
    }

    private static getBroadcastStrides(shape: number[], strides: number[], outShape: number[]): number[] {
        const ndim = outShape.length;
        const ndimIn = shape.length;
        const outStrides = new Array(ndim).fill(0);
        
        for (let i = 0; i < ndim; i++) {
            const dimIn = i - (ndim - ndimIn);
            
            if (dimIn >= 0) {
                if (shape[dimIn] === 1) {
                    outStrides[i] = 0;
                } else {
                    outStrides[i] = strides[dimIn];
                }
            } else {
                outStrides[i] = 0;
            }
        }
        return outStrides;
    }
}
