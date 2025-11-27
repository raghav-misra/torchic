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
    endRow: number
) {
    // Naive implementation with row sharding
    // TODO: Tiling/Blocking for cache efficiency
    for (let i = startRow; i < endRow; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}
