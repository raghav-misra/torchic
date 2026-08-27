// Row-wise causal-masked softmax over an m x n matrix, numerically stable.
// Row r may only attend to columns [0, past_len + r] (inclusive), clamped to n-1.
// Columns beyond that get 0. past_len supports prefill continuation on top of
// an existing KV cache (past_len=0 for from-scratch prefill).
#[no_mangle]
pub unsafe extern "C" fn causal_softmax2d(
    input: *const f32,
    output: *mut f32,
    _m: u32,
    n: u32,
    past_len: u32,
    start_row: u32,
    end_row: u32,
) {
    let n = n as usize;
    let past_len = past_len as usize;
    let start_row = start_row as usize;
    let end_row = end_row as usize;

    for r in start_row..end_row {
        let base = r * n;
        let mut allowed = past_len + r;
        if allowed >= n {
            allowed = n - 1;
        }
        let end_col = allowed + 1;

        let mut maxv = f32::NEG_INFINITY;
        for c in 0..end_col {
            let v = *input.add(base + c);
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0f32;
        for c in 0..end_col {
            let e = libm::expf(*input.add(base + c) - maxv);
            *output.add(base + c) = e;
            sum += e;
        }
        if sum != 0.0 {
            let inv = 1.0 / sum;
            for c in 0..end_col {
                let v = *output.add(base + c);
                *output.add(base + c) = v * inv;
            }
        } else {
            let v = 1.0 / end_col as f32;
            for c in 0..end_col {
                *output.add(base + c) = v;
            }
        }
        for c in end_col..n {
            *output.add(base + c) = 0.0;
        }
    }
}
