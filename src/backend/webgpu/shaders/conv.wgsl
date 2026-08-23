// 1-D convolution forward. Gather formulation so we can do atomic-free
// per-output writes. See workers/kernels/conv.ts for the reference algorithm.
//
// Layout matches PyTorch:
//   input  [B, C_in,  L_in]  (contiguous)
//   weight [C_out, C_in, K]  (contiguous)
//   bias   [C_out]           (contiguous, optional)
//   output [B, C_out, L_out] (contiguous)

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

  let in_b_off = b * u.Cin * u.Lin;
  let w_co_off = co * u.Cin * u.K;

  for (var ci: u32 = 0u; ci < u.Cin; ci = ci + 1u) {
    let in_c_off = in_b_off + ci * u.Lin;
    let w_ci_off = w_co_off + ci * u.K;
    for (var k: u32 = 0u; k < u.K; k = k + 1u) {
      let li = lo_i * u.stride + i32(k) * u.dil - u.pad;
      if (li >= 0 && li < i32(u.Lin)) {
        sum = sum + heap[u.input_off + in_c_off + u32(li)] * heap[u.weight_off + w_ci_off + k];
      }
    }
  }

  heap[u.output_off + idx] = sum;
}

// ConvTranspose1d forward via gather. For each output element (b, co, lo):
//   sum over ci, k of input[b, ci, (lo + pad - k*dil) / stride]  (when divisible)
//                    * weight[ci, co, k]
// weight layout: [C_in, C_out, K]
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

  let in_b_off = b * u.Cin * u.Lin;
  let w_co_stride = u.K;
  let w_ci_stride = u.Cout * u.K;

  for (var ci: u32 = 0u; ci < u.Cin; ci = ci + 1u) {
    let in_c_off = in_b_off + ci * u.Lin;
    let w_ci_off = ci * w_ci_stride + co * w_co_stride;
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
