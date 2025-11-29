export function transpose(
    input: Float32Array,
    output: Float32Array,
    m: number,
    n: number,
    startRow: number,
    endRow: number
) {
    // We are computing rows startRow to endRow of the OUTPUT matrix.
    // The output matrix has shape [n, m].
    // So we iterate r from startRow to endRow (where r is a row in output, col in input).
    
    for (let r = startRow; r < endRow; r++) {
        for (let c = 0; c < m; c++) {
            // output[r, c] = input[c, r]
            output[r * m + c] = input[c * n + r];
        }
    }
}
