// Fused LSTM inference step. One dispatch per timestep instead of the ~20
// composed dispatches (2 matmuls + 3 adds + 4 slices + 4 activations + 3 muls
// + 2 adds). Cuts dispatch overhead on Kokoro's DurationEncoder + shared
// LSTM stack by roughly 15×.
//
// PyTorch gate order (i, f, g, o) — matches nn.LSTM. Weights loaded as
// weight_ih [4H, IN] and weight_hh [4H, H], row-major.
//
// Output is a single [B, 2H] tensor packing [h_new || c_new] along the last
// dim; caller slices into two views.

struct LstmU {
  x_off: u32,
  h_off: u32,
  c_off: u32,
  w_ih_off: u32,
  w_hh_off: u32,
  b_ih_off: u32,
  b_hh_off: u32,
  h_new_off: u32,
  c_new_off: u32,
  batch_size: u32,
  hidden: u32,
  in_size: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: LstmU;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let H = u.hidden;
  let IN = u.in_size;
  let total = u.batch_size * H;
  if (idx >= total) { return; }

  let b = idx / H;
  let k = idx % H;

  let x_base = u.x_off + b * IN;
  let h_base = u.h_off + b * H;
  let c_base = u.c_off + b * H;
  let h_out_base = u.h_new_off + b * H;
  let c_out_base = u.c_new_off + b * H;

  var pre_i: f32 = heap[u.b_ih_off + 0u * H + k] + heap[u.b_hh_off + 0u * H + k];
  var pre_f: f32 = heap[u.b_ih_off + 1u * H + k] + heap[u.b_hh_off + 1u * H + k];
  var pre_g: f32 = heap[u.b_ih_off + 2u * H + k] + heap[u.b_hh_off + 2u * H + k];
  var pre_o: f32 = heap[u.b_ih_off + 3u * H + k] + heap[u.b_hh_off + 3u * H + k];

  let w_ih_i = u.w_ih_off + (0u * H + k) * IN;
  let w_ih_f = u.w_ih_off + (1u * H + k) * IN;
  let w_ih_g = u.w_ih_off + (2u * H + k) * IN;
  let w_ih_o = u.w_ih_off + (3u * H + k) * IN;
  for (var i: u32 = 0u; i < IN; i = i + 1u) {
    let xi = heap[x_base + i];
    pre_i = pre_i + xi * heap[w_ih_i + i];
    pre_f = pre_f + xi * heap[w_ih_f + i];
    pre_g = pre_g + xi * heap[w_ih_g + i];
    pre_o = pre_o + xi * heap[w_ih_o + i];
  }

  let w_hh_i = u.w_hh_off + (0u * H + k) * H;
  let w_hh_f = u.w_hh_off + (1u * H + k) * H;
  let w_hh_g = u.w_hh_off + (2u * H + k) * H;
  let w_hh_o = u.w_hh_off + (3u * H + k) * H;
  for (var j: u32 = 0u; j < H; j = j + 1u) {
    let hj = heap[h_base + j];
    pre_i = pre_i + hj * heap[w_hh_i + j];
    pre_f = pre_f + hj * heap[w_hh_f + j];
    pre_g = pre_g + hj * heap[w_hh_g + j];
    pre_o = pre_o + hj * heap[w_hh_o + j];
  }

  let ig = 1.0 / (1.0 + exp(-pre_i));
  let fg = 1.0 / (1.0 + exp(-pre_f));
  let gc = tanh(pre_g);
  let og = 1.0 / (1.0 + exp(-pre_o));

  let c_prev = heap[c_base + k];
  let c_new = fg * c_prev + ig * gc;
  let h_new = og * tanh(c_new);

  heap[h_out_base + k] = h_new;
  heap[c_out_base + k] = c_new;
}
