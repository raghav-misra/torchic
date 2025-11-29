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
    // Improved cache-friendly implementation (i, p, j ordering).
    // Zero each output row then accumulate across k so accesses to B are
    // row-major (contiguous) when bRowStride/bColStride are default.
    // Supports optional strides for A and B (to handle views without materialize).
    const aRowStride = stridesA ? stridesA[0] : k;
    const aColStride = stridesA ? stridesA[1] : 1;
    const bRowStride = stridesB ? stridesB[0] : n;
    const bColStride = stridesB ? stridesB[1] : 1;

    for (let i = startRow; i < endRow; i++) {
        const aRowBase = i * aRowStride;
        const outRowBase = i * n;

        // Initialize output row to zero
        for (let j = 0; j < n; j++) out[outRowBase + j] = 0;

        // Accumulate: for each k (p) add a[i,p] * B[p, :]
        for (let p = 0; p < k; p++) {
            const aVal = a[aRowBase + p * aColStride];
            const bRowBase = p * bRowStride;

            // simple 4-wide unroll to help the engine emit faster code
            let j = 0;
            const limit = n - 3;
            for (; j <= limit; j += 4) {
                const b0 = b[bRowBase + (j + 0) * bColStride];
                const b1 = b[bRowBase + (j + 1) * bColStride];
                const b2 = b[bRowBase + (j + 2) * bColStride];
                const b3 = b[bRowBase + (j + 3) * bColStride];

                out[outRowBase + j + 0] += aVal * b0;
                out[outRowBase + j + 1] += aVal * b1;
                out[outRowBase + j + 2] += aVal * b2;
                out[outRowBase + j + 3] += aVal * b3;
            }
            for (; j < n; j++) {
                out[outRowBase + j] += aVal * b[bRowBase + j * bColStride];
            }
        }
    }
}
