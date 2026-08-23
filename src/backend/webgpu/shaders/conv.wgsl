// 1-D conv forward with grouped-conv support. Gather formulation — no atomics.
// Layout matches PyTorch:
//   input  [B, C_in, L_in]
//   weight [C_out, C_in/G, K]  for conv1d,  [C_in, C_out/G, K]  for conv_transpose1d
//   bias   [C_out] (optional)
//   output [B, C_out, L_out]

struct ConvU {
  input_off: u32,
  weight_off: u32,
  bias_off: u32,
  output_off: u32,
  has_bias: u32,
  B: u32,
  Cin: u32,
  Lin: u32,
  Cout: u32,
  K: u32,
  Lout: u32,
  stride: i32,
  pad: i32,
  dil: i32,
  groups: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: ConvU;

@compute @workgroup_size(64)
fn conv1d(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = u.B * u.Cout * u.Lout;
  let idx = gid.x;
  if (idx >= total) { return; }

  let lo_i = i32(idx % u.Lout);
  let tmp = idx / u.Lout;
  let co = tmp % u.Cout;
  let b = tmp / u.Cout;

  var sum: f32 = 0.0;
  if (u.has_bias != 0u) { sum = heap[u.bias_off + co]; }

  let cin_per_g = u.Cin / u.groups;
  let cout_per_g = u.Cout / u.groups;
  let g = co / cout_per_g;
  let ci_start = g * cin_per_g;

  let in_b_off = b * u.Cin * u.Lin;
  let w_co_off = co * cin_per_g * u.K;

  for (var ci_off: u32 = 0u; ci_off < cin_per_g; ci_off = ci_off + 1u) {
    let ci = ci_start + ci_off;
    let in_c_off = in_b_off + ci * u.Lin;
    let w_ci_off = w_co_off + ci_off * u.K;
    for (var k: u32 = 0u; k < u.K; k = k + 1u) {
      let li = lo_i * u.stride + i32(k) * u.dil - u.pad;
      if (li >= 0 && li < i32(u.Lin)) {
        sum = sum + heap[u.input_off + in_c_off + u32(li)] * heap[u.weight_off + w_ci_off + k];
      }
    }
  }

  heap[u.output_off + idx] = sum;
}

@compute @workgroup_size(64)
fn conv_transpose1d(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = u.B * u.Cout * u.Lout;
  let idx = gid.x;
  if (idx >= total) { return; }

  let lo_i = i32(idx % u.Lout);
  let tmp = idx / u.Lout;
  let co = tmp % u.Cout;
  let b = tmp / u.Cout;

  var sum: f32 = 0.0;
  if (u.has_bias != 0u) { sum = heap[u.bias_off + co]; }

  let cin_per_g = u.Cin / u.groups;
  let cout_per_g = u.Cout / u.groups;
  let g = co / cout_per_g;
  let ci_start = g * cin_per_g;
  let co_off = co - g * cout_per_g;

  let in_b_off = b * u.Cin * u.Lin;
  let w_co_stride = u.K;
  let w_ci_stride = cout_per_g * u.K;

  for (var ci_off: u32 = 0u; ci_off < cin_per_g; ci_off = ci_off + 1u) {
    let ci = ci_start + ci_off;
    let in_c_off = in_b_off + ci * u.Lin;
    let w_ci_off = ci * w_ci_stride + co_off * w_co_stride;
    for (var k: u32 = 0u; k < u.K; k = k + 1u) {
      let j = lo_i + u.pad - i32(k) * u.dil;
      if (j >= 0 && (j % u.stride) == 0) {
        let li = j / u.stride;
        if (li >= 0 && li < i32(u.Lin)) {
          sum = sum + heap[u.input_off + in_c_off + u32(li)] * heap[u.weight_off + w_ci_off + k];
        }
      }
    }
  }

  heap[u.output_off + idx] = sum;
}
