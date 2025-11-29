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
    // Blocked/tiled matrix multiplication for improved cache locality and scaling
    const BLOCK_SIZE = 32;
    const aRowStride = stridesA ? stridesA[0] : k;
    const aColStride = stridesA ? stridesA[1] : 1;
    const bRowStride = stridesB ? stridesB[0] : n;
    const bColStride = stridesB ? stridesB[1] : 1;

    for (let i0 = startRow; i0 < endRow; i0 += BLOCK_SIZE) {
        const iMax = Math.min(i0 + BLOCK_SIZE, endRow);
        for (let j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            const jMax = Math.min(j0 + BLOCK_SIZE, n);
            for (let p0 = 0; p0 < k; p0 += BLOCK_SIZE) {
                const pMax = Math.min(p0 + BLOCK_SIZE, k);
                // For each block, do standard matmul
                for (let i = i0; i < iMax; i++) {
                    const aRowBase = i * aRowStride;
                    const outRowBase = i * n;
                    // Only zero output row once per block
                    if (p0 === 0) {
                        for (let j = j0; j < jMax; j++) {
                            out[outRowBase + j] = 0;
                        }
                    }
                    for (let p = p0; p < pMax; p++) {
                        const aVal = a[aRowBase + p * aColStride];
                        const bRowBase = p * bRowStride;
                        for (let j = j0; j < jMax; j++) {
                            out[outRowBase + j] += aVal * b[bRowBase + j * bColStride];
                        }
                    }
                }
            }
        }
    }
}
