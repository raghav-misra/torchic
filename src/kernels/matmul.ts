// Matrix multiplication kernel
// C = A * B
// A: M x K
// B: K x N
// C: M x N
export function matmul(
    a: Float32Array,
    b: Float32Array,
    out: Float32Array,
    m: number,
    n: number,
    k: number,
    startRow: number,
    endRow: number,
    stridesA?: number[],
    stridesB?: number[]
) {
    // Naive implementation with row sharding
    // Supports optional strides for A and B (to handle views without materialize)
    const aRowStride = stridesA ? stridesA[0] : k;
    const aColStride = stridesA ? stridesA[1] : 1;
    const bRowStride = stridesB ? stridesB[0] : n;
    const bColStride = stridesB ? stridesB[1] : 1;

    for (let i = startRow; i < endRow; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            const aRowBase = i * aRowStride;
            const outBase = i * n + j;
            for (let p = 0; p < k; p++) {
                const aVal = a[aRowBase + p * aColStride];
                const bVal = b[p * bRowStride + j * bColStride];
                sum += aVal * bVal;
            }
            out[outBase] = sum;
        }
    }
}
