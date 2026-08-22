export function transpose(
  input: Float32Array,
  output: Float32Array,
  m: number,
  n: number,
  startRow: number,
  endRow: number,
) {
  for (let r = startRow; r < endRow; r++) {
    for (let c = 0; c < m; c++) {
      output[r * m + c] = input[c * n + r];
    }
  }
}
