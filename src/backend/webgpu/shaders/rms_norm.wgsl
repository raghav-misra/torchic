// Row-parallel 2D RMSNorm. One thread per row.
//   rms = sqrt(mean(x^2) + eps)
//   y[r, c] = x[r, c] / rms * weight[c]

struct RmsNormU {
  input: u32,
  weight: u32,
  output: u32,
  m: u32,
  n: u32,
  eps: f32,
  start_row: u32,
  end_row: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: RmsNormU;

@compute @workgroup_size(64)
fn rms_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = u.start_row + gid.x;
  if (row >= u.end_row) { return; }
  let in_base = u.input + row * u.n;
  let out_base = u.output + row * u.n;

  var sumsq: f32 = 0.0;
  for (var c: u32 = 0u; c < u.n; c = c + 1u) {
    let v = heap[in_base + c];
    sumsq = sumsq + v * v;
  }

  let inv_rms = 1.0 / sqrt(sumsq / f32(u.n) + u.eps);
  for (var c: u32 = 0u; c < u.n; c = c + 1u) {
    let w = heap[u.weight + c];
    heap[out_base + c] = heap[in_base + c] * inv_rms * w;
  }
}
