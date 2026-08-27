// Row-parallel RoPE (half-split, HF Llama convention).
// One thread per row of a [total_rows, D] view where row r maps to time r % t_seq.
// cos/sin tables are [t_seq, D/2].
//   x'[i]         = x[i] * cos[t, i] - x[i + D/2] * sin[t, i]
//   x'[i + D/2]   = x[i] * sin[t, i] + x[i + D/2] * cos[t, i]

struct RopeU {
  x: u32,
  cos: u32,
  sin: u32,
  out: u32,
  t_seq: u32,
  d_half: u32,
  start_row: u32,
  end_row: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: RopeU;

@compute @workgroup_size(64)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = u.start_row + gid.x;
  if (row >= u.end_row) { return; }
  let d = 2u * u.d_half;
  let time = row % u.t_seq;
  let x_base = u.x + row * d;
  let out_base = u.out + row * d;
  let cs_base = time * u.d_half;
  for (var i: u32 = 0u; i < u.d_half; i = i + 1u) {
    let a = heap[x_base + i];
    let b = heap[x_base + i + u.d_half];
    let c = heap[u.cos + cs_base + i];
    let s = heap[u.sin + cs_base + i];
    heap[out_base + i] = a * c - b * s;
    heap[out_base + i + u.d_half] = a * s + b * c;
  }
}
