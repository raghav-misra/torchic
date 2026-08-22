export function sum_partial(
  input: Float32Array,
  output: Float32Array,
  outIndex: number,
  start: number,
  end: number,
) {
  let sum = 0;
  for (let i = start; i < end; i++) {
    sum += input[i];
  }
  output[outIndex] = sum;
}

export function sum_final(input: Float32Array, output: Float32Array, n: number) {
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += input[i];
  }
  output[0] = sum;
}

export function sum_axis(
  input: Float32Array,
  output: Float32Array,
  start: number,
  end: number,
  shape: number[],
  strides: number[],
  axis: number,
) {
  const dimSize = shape[axis];
  const dimStride = strides[axis];

  // (stride, size) for the non-reduction dims, innermost first, so decomposing
  // a flat output index i gives the corresponding input offset.
  const outDims: { stride: number; size: number }[] = [];
  for (let d = shape.length - 1; d >= 0; d--) {
    if (d === axis) continue;
    outDims.push({ stride: strides[d], size: shape[d] });
  }

  for (let i = start; i < end; i++) {
    let inputOffset = 0;
    let rem = i;

    for (const dim of outDims) {
      const coord = rem % dim.size;
      rem = Math.floor(rem / dim.size);
      inputOffset += coord * dim.stride;
    }

    let sum = 0;
    for (let k = 0; k < dimSize; k++) {
      sum += input[inputOffset + k * dimStride];
    }
    output[i] = sum;
  }
}

export function add_scalar_tensor(
  a: Float32Array,
  scalar: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
) {
  const val = scalar[0];
  for (let i = start; i < end; i++) {
    out[i] = a[i] + val;
  }
}
