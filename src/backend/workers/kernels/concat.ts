// Concat: copy a contiguous input tensor into a strided slab of the output.
// Same shape as the WASM crate's concat_slab. Called once per input tensor.

export function concat_slab(
  input: Float32Array,
  out: Float32Array,
  outerSize: number,
  inAxisSize: number,
  outAxisSize: number,
  axisOffset: number,
  innerSize: number,
  start: number,
  end: number,
) {
  const axisXInner = inAxisSize * innerSize;
  const outAxisXInner = outAxisSize * innerSize;
  for (let idx = start; idx < end; idx++) {
    const outer = Math.floor(idx / axisXInner);
    const rem = idx - outer * axisXInner;
    const axis = Math.floor(rem / innerSize);
    const inner = rem - axis * innerSize;
    out[outer * outAxisXInner + (axisOffset + axis) * innerSize + inner] = input[idx];
  }
  void outerSize;
}
