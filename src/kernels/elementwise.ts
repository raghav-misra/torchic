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
export function relu(a: Float32Array, out: Float32Array, start: number, end: number, shape?: number[], strides?: number[]) {
    if (shape && strides) {
        // strided access: map flat output index -> input offset
        for (let i = start; i < end; i++) {
            let idx = i;
            let inputOffset = 0;
            for (let dim = shape.length - 1; dim >= 0; dim--) {
                const size = shape[dim];
                const pos = idx % size;
                idx = Math.floor(idx / size);
                inputOffset += pos * strides[dim];
            }
            out[i] = Math.max(0, a[inputOffset]);
        }
    } else {
        for (let i = start; i < end; i++) {
            out[i] = Math.max(0, a[i]);
        }
    }
}

export function exp(a: Float32Array, out: Float32Array, start: number, end: number, shape?: number[], strides?: number[]) {
    if (shape && strides) {
        for (let i = start; i < end; i++) {
            let idx = i;
            let inputOffset = 0;
            for (let dim = shape.length - 1; dim >= 0; dim--) {
                const size = shape[dim];
                const pos = idx % size;
                idx = Math.floor(idx / size);
                inputOffset += pos * strides[dim];
            }
            out[i] = Math.exp(a[inputOffset]);
        }
    } else {
        for (let i = start; i < end; i++) {
            out[i] = Math.exp(a[i]);
        }
    }
}

export function log(a: Float32Array, out: Float32Array, start: number, end: number, shape?: number[], strides?: number[]) {
    if (shape && strides) {
        for (let i = start; i < end; i++) {
            let idx = i;
            let inputOffset = 0;
            for (let dim = shape.length - 1; dim >= 0; dim--) {
                const size = shape[dim];
                const pos = idx % size;
                idx = Math.floor(idx / size);
                inputOffset += pos * strides[dim];
            }
            out[i] = Math.log(a[inputOffset]);
        }
    } else {
        for (let i = start; i < end; i++) {
            out[i] = Math.log(a[i]);
        }
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

export function materialize(input: Float32Array, out: Float32Array, start: number, end: number, shape: number[], strides: number[]) {
    // Convert non-contiguous tensor to contiguous
    for (let i = start; i < end; i++) {
        // Map flat output index to strided input offset
        let inputOffset = 0;
        let idx = i;
        for (let dim = shape.length - 1; dim >= 0; dim--) {
            const pos = idx % shape[dim];
            idx = Math.floor(idx / shape[dim]);
            inputOffset += pos * strides[dim];
        }
        out[i] = input[inputOffset];
    }
}

export function relu_backward(input: Float32Array, gradOutput: Float32Array, gradInput: Float32Array, start: number, end: number, shape?: number[], strides?: number[]) {
    if (shape && strides) {
        for (let i = start; i < end; i++) {
            let idx = i;
            let inputOffset = 0;
            for (let dim = shape.length - 1; dim >= 0; dim--) {
                const size = shape[dim];
                const pos = idx % size;
                idx = Math.floor(idx / size);
                inputOffset += pos * strides[dim];
            }
            gradInput[i] = input[inputOffset] > 0 ? gradOutput[i] : 0;
        }
    } else {
        for (let i = start; i < end; i++) {
            gradInput[i] = input[i] > 0 ? gradOutput[i] : 0;
        }
    }
}

export function tanh(a: Float32Array, out: Float32Array, start: number, end: number, shape?: number[], strides?: number[]) {
    if (shape && strides) {
        for (let i = start; i < end; i++) {
            let idx = i;
            let inputOffset = 0;
            for (let dim = shape.length - 1; dim >= 0; dim--) {
                const size = shape[dim];
                const pos = idx % size;
                idx = Math.floor(idx / size);
                inputOffset += pos * strides[dim];
            }
            out[i] = Math.tanh(a[inputOffset]);
        }
    } else {
        for (let i = start; i < end; i++) {
            out[i] = Math.tanh(a[i]);
        }
    }
}

export function tanh_backward(output: Float32Array, gradOutput: Float32Array, gradInput: Float32Array, start: number, end: number) {
    // output is tanh(input); derivative = 1 - output^2
    for (let i = start; i < end; i++) {
        const o = output[i];
        gradInput[i] = gradOutput[i] * (1 - o * o);
    }
}

// Softmax optimized for 2D tensors on the last axis (rows are independent)
export function softmax2d(input: Float32Array, out: Float32Array, m: number, n: number, startRow: number, endRow: number) {
    for (let r = startRow; r < endRow; r++) {
        const base = r * n;
        let maxv = -Infinity;
        for (let c = 0; c < n; c++) {
            const v = input[base + c];
            if (v > maxv) maxv = v;
        }
        let sum = 0.0;
        for (let c = 0; c < n; c++) {
            const e = Math.exp(input[base + c] - maxv);
            out[base + c] = e;
            sum += e;
        }
        // normalize
        if (sum !== 0) {
            const inv = 1.0 / sum;
            for (let c = 0; c < n; c++) out[base + c] = out[base + c] * inv;
        } else {
            const v = 1.0 / n;
            for (let c = 0; c < n; c++) out[base + c] = v;
        }
    }
}

export function softmax_backward2d(output: Float32Array, gradOutput: Float32Array, gradInput: Float32Array, m: number, n: number, startRow: number, endRow: number) {
    for (let r = startRow; r < endRow; r++) {
        const base = r * n;
        let dot = 0.0;
        for (let c = 0; c < n; c++) {
            dot += gradOutput[base + c] * output[base + c];
        }
        for (let c = 0; c < n; c++) {
            gradInput[base + c] = output[base + c] * (gradOutput[base + c] - dot);
        }
    }
}
