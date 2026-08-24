// Fused Snake activation: y = x + (1/α) * sin²(α·x). α broadcasts along B and T
// so we index it by channel only.
// x:     [B, C, T] contiguous
// alpha: [C]       (caller flattens the [1,C,1] layout)
// out:   [B, C, T]

struct SnakeU {
  input_off: u32,
  alpha_off: u32,
  output_off: u32,
  numel: u32,
  channels: u32,
  inner: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: SnakeU;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.numel) { return; }
  let c = (i / u.inner) % u.channels;
  let x = heap[u.input_off + i];
  let a = heap[u.alpha_off + c];
  let s = sin(a * x);
  heap[u.output_off + i] = x + s * s / a;
}
