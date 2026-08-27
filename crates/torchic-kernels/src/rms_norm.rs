use core::arch::wasm32::{f32x4_extract_lane, f32x4_splat, f32x4_sqrt};

// Row-wise RMSNorm over an m x n matrix. Each worker handles rows [start_row, end_row).
//   rms = sqrt(mean(x^2) + eps)
//   y[r, c] = x[r, c] / rms * weight[c]
// weight is a length-n vector applied per column.
//
// sqrt goes through the WASM native f32x4.sqrt intrinsic (hardware sqrt).
// libm::sqrtf is a soft polynomial impl whose coefficient LUT lives inside
// the module's data section (~offset 1048848), which torchic's tensor heap
// starts on top of — a call to libm::sqrtf here reads garbage LUT bytes and
// returns wrong values. Same reason elementwise::sqrt_op uses f32x4_sqrt.
#[no_mangle]
pub unsafe extern "C" fn rms_norm2d(
    input: *const f32,
    weight: *const f32,
    output: *mut f32,
    _m: u32,
    n: u32,
    eps: f32,
    start_row: u32,
    end_row: u32,
) {
    let n = n as usize;
    let start_row = start_row as usize;
    let end_row = end_row as usize;
    let inv_n = 1.0f32 / n as f32;

    for r in start_row..end_row {
        let base = r * n;
        let mut sumsq = 0.0f32;
        for c in 0..n {
            let v = *input.add(base + c);
            sumsq += v * v;
        }
        let rms = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(sumsq * inv_n + eps)));
        let inv_rms = 1.0f32 / rms;
        for c in 0..n {
            let v = *input.add(base + c);
            let w = *weight.add(c);
            *output.add(base + c) = v * inv_rms * w;
        }
    }
}
