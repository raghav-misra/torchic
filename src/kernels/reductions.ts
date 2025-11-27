export function sum_partial(
    input: Float32Array,
    output: Float32Array,
    outIndex: number,
    start: number,
    end: number
) {
    let sum = 0;
    for (let i = start; i < end; i++) {
        sum += input[i];
    }
    output[outIndex] = sum;
}

export function sum_final(
    input: Float32Array,
    output: Float32Array,
    n: number
) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += input[i];
    }
    output[0] = sum;
}

export function add_scalar_tensor(
    a: Float32Array,
    scalar: Float32Array,
    out: Float32Array,
    start: number,
    end: number
) {
    const val = scalar[0];
    for (let i = start; i < end; i++) {
        out[i] = a[i] + val;
    }
}
