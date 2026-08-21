// Standard-normal generator via Box-Muller with a per-thread xorshift32.
// JS-side passes a base seed; each thread mixes it with its global id to get
// independent streams.

struct RandnU {
  output_off: u32,
  count: u32,
  seed: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: RandnU;

fn xorshift32(state: ptr<function, u32>) -> u32 {
  var x = *state;
  x = x ^ (x << 13u);
  x = x ^ (x >> 17u);
  x = x ^ (x << 5u);
  *state = x;
  return x;
}

fn next_unit(state: ptr<function, u32>) -> f32 {
  // (0, 1] so log() stays finite. 4294967297.0 loses a few bits in f32 but is
  // fine for RNG quality.
  return (f32(xorshift32(state)) + 1.0) / 4294967297.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  var state: u32 = u.seed ^ (i * 0x9E3779B1u);
  if (state == 0u) { state = 0x9E3779B9u; }
  let a = next_unit(&state);
  let b = next_unit(&state);
  let two_pi = 6.283185307179586;
  heap[u.output_off + i] = sqrt(-2.0 * log(a)) * cos(two_pi * b);
}
