// Fused LSTM inference step (WASM backend). Mirrors
// src/backend/webgpu/shaders/lstm_step.wgsl and
// src/backend/workers/kernels/lstm.ts. Gate order i, f, g, o; weight_ih
// is [4H, IN], weight_hh is [4H, H]. Output packs [h_new || c_new] along
// the last dim.

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

#[no_mangle]
pub unsafe extern "C" fn lstm_step(
    x: *const f32,
    h: *const f32,
    c: *const f32,
    w_ih: *const f32,
    w_hh: *const f32,
    b_ih: *const f32,
    b_hh: *const f32,
    out: *mut f32,
    batch_size: u32,
    hidden: u32,
    in_size: u32,
) {
    let batch_size = batch_size as usize;
    let h_sz = hidden as usize;
    let in_sz = in_size as usize;

    for b in 0..batch_size {
        let x_base = b * in_sz;
        let h_base = b * h_sz;
        let c_base = b * h_sz;
        let out_base = b * 2 * h_sz;

        for k in 0..h_sz {
            let mut pre_i = *b_ih.add(0 * h_sz + k) + *b_hh.add(0 * h_sz + k);
            let mut pre_f = *b_ih.add(1 * h_sz + k) + *b_hh.add(1 * h_sz + k);
            let mut pre_g = *b_ih.add(2 * h_sz + k) + *b_hh.add(2 * h_sz + k);
            let mut pre_o = *b_ih.add(3 * h_sz + k) + *b_hh.add(3 * h_sz + k);

            let row_i_ih = (0 * h_sz + k) * in_sz;
            let row_f_ih = (1 * h_sz + k) * in_sz;
            let row_g_ih = (2 * h_sz + k) * in_sz;
            let row_o_ih = (3 * h_sz + k) * in_sz;
            for i in 0..in_sz {
                let xi = *x.add(x_base + i);
                pre_i += xi * *w_ih.add(row_i_ih + i);
                pre_f += xi * *w_ih.add(row_f_ih + i);
                pre_g += xi * *w_ih.add(row_g_ih + i);
                pre_o += xi * *w_ih.add(row_o_ih + i);
            }

            let row_i_hh = (0 * h_sz + k) * h_sz;
            let row_f_hh = (1 * h_sz + k) * h_sz;
            let row_g_hh = (2 * h_sz + k) * h_sz;
            let row_o_hh = (3 * h_sz + k) * h_sz;
            for j in 0..h_sz {
                let hj = *h.add(h_base + j);
                pre_i += hj * *w_hh.add(row_i_hh + j);
                pre_f += hj * *w_hh.add(row_f_hh + j);
                pre_g += hj * *w_hh.add(row_g_hh + j);
                pre_o += hj * *w_hh.add(row_o_hh + j);
            }

            let ig = sigmoid(pre_i);
            let fg = sigmoid(pre_f);
            let gc = libm::tanhf(pre_g);
            let og = sigmoid(pre_o);

            let c_prev = *c.add(c_base + k);
            let c_new = fg * c_prev + ig * gc;
            let h_new = og * libm::tanhf(c_new);

            *out.add(out_base + k) = h_new;
            *out.add(out_base + h_sz + k) = c_new;
        }
    }
}
