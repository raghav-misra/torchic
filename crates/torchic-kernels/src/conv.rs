// 1-D convolution forward-only kernels. Input layout [B, C_in, L_in],
// weight layout [C_out, C_in/G, K], output [B, C_out, L_out]. G = groups.
// Same math as workers/kernels/conv.ts; scalar-only for now, SIMD pass later.

// Conv1d: L_out = floor((L_in + 2*pad - dil*(K-1) - 1) / stride) + 1
#[no_mangle]
pub unsafe extern "C" fn conv1d(
    input: *const f32,
    weight: *const f32,
    bias: *const f32, // null when has_bias == 0
    out: *mut f32,
    has_bias: u32,
    _b_total: u32,
    c_in: u32,
    l_in: u32,
    c_out: u32,
    k: u32,
    l_out: u32,
    stride: u32,
    pad: u32,
    dil: u32,
    groups: u32,
    start_batch: u32,
    end_batch: u32,
) {
    let c_in = c_in as usize;
    let l_in = l_in as isize;
    let c_out = c_out as usize;
    let k = k as usize;
    let l_out = l_out as usize;
    let stride = stride as isize;
    let pad = pad as isize;
    let dil = dil as isize;
    let groups = groups as usize;
    let start_batch = start_batch as usize;
    let end_batch = end_batch as usize;

    let cin_per_group = c_in / groups;
    let cout_per_group = c_out / groups;

    let in_stride_b = c_in * (l_in as usize);
    let in_stride_c = l_in as usize;
    let w_stride_co = cin_per_group * k;
    let w_stride_ci = k;
    let out_stride_b = c_out * l_out;
    let out_stride_c = l_out;

    for b in start_batch..end_batch {
        let in_b = input.add(b * in_stride_b);
        let out_b = out.add(b * out_stride_b);
        for co in 0..c_out {
            let g = co / cout_per_group;
            let ci_start = g * cin_per_group;
            let out_base = out_b.add(co * out_stride_c);
            let w_co = weight.add(co * w_stride_co);
            let bias_val = if has_bias != 0 { *bias.add(co) } else { 0.0 };
            for lo in 0..l_out {
                let mut sum = bias_val;
                for ci_off in 0..cin_per_group {
                    let ci = ci_start + ci_off;
                    let in_c = in_b.add(ci * in_stride_c);
                    let w_ci = w_co.add(ci_off * w_stride_ci);
                    for kk in 0..k {
                        let li = lo as isize * stride + kk as isize * dil - pad;
                        if li >= 0 && li < l_in {
                            sum += *in_c.add(li as usize) * *w_ci.add(kk);
                        }
                    }
                }
                *out_base.add(lo) = sum;
            }
        }
    }
}

// ConvTranspose1d: L_out = (L_in-1)*stride - 2*pad + dil*(K-1) + output_pad + 1.
// Weight layout [C_in, C_out/G, K]. Init output with bias (or zero), then scatter.
#[no_mangle]
pub unsafe extern "C" fn conv_transpose1d(
    input: *const f32,
    weight: *const f32,
    bias: *const f32,
    out: *mut f32,
    has_bias: u32,
    _b_total: u32,
    c_in: u32,
    l_in: u32,
    c_out: u32,
    k: u32,
    l_out: u32,
    stride: u32,
    pad: u32,
    dil: u32,
    groups: u32,
    start_batch: u32,
    end_batch: u32,
) {
    let c_in = c_in as usize;
    let l_in = l_in as usize;
    let c_out = c_out as usize;
    let k = k as usize;
    let l_out = l_out as usize;
    let stride = stride as isize;
    let pad = pad as isize;
    let dil = dil as isize;
    let groups = groups as usize;
    let start_batch = start_batch as usize;
    let end_batch = end_batch as usize;

    let cin_per_group = c_in / groups;
    let cout_per_group = c_out / groups;

    let in_stride_b = c_in * l_in;
    let in_stride_c = l_in;
    let w_stride_ci = cout_per_group * k;
    let w_stride_co = k;
    let out_stride_b = c_out * l_out;
    let out_stride_c = l_out;
    let l_out_i = l_out as isize;

    for b in start_batch..end_batch {
        let in_b = input.add(b * in_stride_b);
        let out_b = out.add(b * out_stride_b);
        for co in 0..c_out {
            let out_base = out_b.add(co * out_stride_c);
            let biasv = if has_bias != 0 { *bias.add(co) } else { 0.0 };
            for lo in 0..l_out {
                *out_base.add(lo) = biasv;
            }
        }
        for ci in 0..c_in {
            let g = ci / cin_per_group;
            let co_start = g * cout_per_group;
            let in_c = in_b.add(ci * in_stride_c);
            let w_ci = weight.add(ci * w_stride_ci);
            for li in 0..l_in {
                let xv = *in_c.add(li);
                if xv == 0.0 {
                    continue;
                }
                for co_off in 0..cout_per_group {
                    let co = co_start + co_off;
                    let out_base = out_b.add(co * out_stride_c);
                    let w_base = w_ci.add(co_off * w_stride_co);
                    for kk in 0..k {
                        let lo = li as isize * stride + kk as isize * dil - pad;
                        if lo >= 0 && lo < l_out_i {
                            *out_base.add(lo as usize) += xv * *w_base.add(kk);
                        }
                    }
                }
            }
        }
    }
}
