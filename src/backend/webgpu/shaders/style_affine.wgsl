// Fused StyleTTS AdaIN affine: y = x * (1 + gamma) + beta.
// x is [..., C, ...] with `channels` at some interior axis and `inner` = prod
// of trailing dims. gamma/beta are per-channel and index by (i / inner) % C.
// Caller must ensure B == 1 (or fold B*T into inner) and gamma/beta are
// contiguous C-element vectors.

struct AffineU {
  x_off: u32,
  gamma_off: u32,
  beta_off: u32,
  output_off: u32,
  numel: u32,
  channels: u32,
  inner: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: AffineU;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.numel) { return; }
  let c = (i / u.inner) % u.channels;
  let x = heap[u.x_off + i];
  let g = heap[u.gamma_off + c];
  let b = heap[u.beta_off + c];
  heap[u.output_off + i] = x * (1.0 + g) + b;
}
