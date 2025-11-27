// Element-wise operations
// These are embarrassingly parallel. We just split the array into N chunks.

function getOffsets(i: number, shape: number[], stridesA: number[], stridesB: number[]) {
    let idx = i;
    let offsetA = 0;
    let offsetB = 0;
    for (let dim = shape.length - 1; dim >= 0; dim--) {
        const size = shape[dim];
        const pos = idx % size;
        idx = Math.floor(idx / size);
        offsetA += pos * stridesA[dim];
        offsetB += pos * stridesB[dim];
    }
    return [offsetA, offsetB];
}

export function add(a: Float32Array, b: Float32Array, out: Float32Array, start: number, end: number, shape?: number[], stridesA?: number[], stridesB?: number[]) {
    if (shape && stridesA && stridesB) {
        for (let i = start; i < end; i++) {
            const [offA, offB] = getOffsets(i, shape, stridesA, stridesB);
            out[i] = a[offA] + b[offB];
        }
    } else {
        for (let i = start; i < end; i++) {
            out[i] = a[i] + b[i];
        }
    }
}

export function sub(a: Float32Array, b: Float32Array, out: Float32Array, start: number, end: number, shape?: number[], stridesA?: number[], stridesB?: number[]) {
    if (shape && stridesA && stridesB) {
        for (let i = start; i < end; i++) {
            const [offA, offB] = getOffsets(i, shape, stridesA, stridesB);
            out[i] = a[offA] - b[offB];
        }
    } else {
        for (let i = start; i < end; i++) {
            out[i] = a[i] - b[i];
        }
    }
}

export function mul(a: Float32Array, b: Float32Array, out: Float32Array, start: number, end: number, shape?: number[], stridesA?: number[], stridesB?: number[]) {
    if (shape && stridesA && stridesB) {
        for (let i = start; i < end; i++) {
            const [offA, offB] = getOffsets(i, shape, stridesA, stridesB);
            out[i] = a[offA] * b[offB];
        }
    } else {
        for (let i = start; i < end; i++) {
            out[i] = a[i] * b[i];
        }
    }
}

export function div(a: Float32Array, b: Float32Array, out: Float32Array, start: number, end: number, shape?: number[], stridesA?: number[], stridesB?: number[]) {
    if (shape && stridesA && stridesB) {
        for (let i = start; i < end; i++) {
            const [offA, offB] = getOffsets(i, shape, stridesA, stridesB);
            out[i] = a[offA] / b[offB];
        }
    } else {
        for (let i = start; i < end; i++) {
            out[i] = a[i] / b[i];
        }
    }
}

// Unary ops
export function relu(a: Float32Array, out: Float32Array, start: number, end: number) {
    for (let i = start; i < end; i++) {
        out[i] = Math.max(0, a[i]);
    }
}

export function exp(a: Float32Array, out: Float32Array, start: number, end: number) {
    for (let i = start; i < end; i++) {
        out[i] = Math.exp(a[i]);
    }
}

export function log(a: Float32Array, out: Float32Array, start: number, end: number) {
    for (let i = start; i < end; i++) {
        out[i] = Math.log(a[i]);
    }
}

export function fill(out: Float32Array, val: number, start: number, end: number) {
    // Native fill is faster, but we need to respect the range
    out.fill(val, start, end);
}

export function randn(out: Float32Array, start: number, end: number) {
    for (let i = start; i < end; i++) {
        // Box-Muller transform for Gaussian distribution
        const u = 1 - Math.random();
        const v = Math.random();
        out[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}

export function copy(input: Float32Array, out: Float32Array, start: number, end: number) {
    for (let i = start; i < end; i++) {
        out[i] = input[i];
    }
}

export function relu_backward(input: Float32Array, gradOutput: Float32Array, gradInput: Float32Array, start: number, end: number) {
    for (let i = start; i < end; i++) {
        gradInput[i] = input[i] > 0 ? gradOutput[i] : 0;
    }
}
