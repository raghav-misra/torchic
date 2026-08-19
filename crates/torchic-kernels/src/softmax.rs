// Row-wise softmax over an m x n matrix. Each worker handles rows [start_row, end_row).
// Numerically stable via max-subtraction.
#[no_mangle]
pub unsafe extern "C" fn softmax2d(
    input: *const f32,
    output: *mut f32,
    _m: u32,
    n: u32,
    start_row: u32,
    end_row: u32,
) {
    let n = n as usize;
    let start_row = start_row as usize;
    let end_row = end_row as usize;

    for r in start_row..end_row {
        let base = r * n;
        let mut maxv = f32::NEG_INFINITY;
        for c in 0..n {
            let v = *input.add(base + c);
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0f32;
        for c in 0..n {
            let e = libm::expf(*input.add(base + c) - maxv);
            *output.add(base + c) = e;
            sum += e;
        }
        if sum != 0.0 {
            let inv = 1.0 / sum;
            for c in 0..n {
                let v = *output.add(base + c);
                *output.add(base + c) = v * inv;
            }
        } else {
            let v = 1.0 / n as f32;
            for c in 0..n {
                *output.add(base + c) = v;
            }
        }
    }
}

// grad_input[r, c] = output[r, c] * (grad_output[r, c] - <grad_output[r, :], output[r, :]>)
#[no_mangle]
pub unsafe extern "C" fn softmax_backward2d(
    output: *const f32,
    grad_output: *const f32,
    grad_input: *mut f32,
    _m: u32,
    n: u32,
    start_row: u32,
    end_row: u32,
) {
    let n = n as usize;
    let start_row = start_row as usize;
    let end_row = end_row as usize;

    for r in start_row..end_row {
        let base = r * n;
        let mut dot = 0.0f32;
        for c in 0..n {
            dot += *grad_output.add(base + c) * *output.add(base + c);
        }
        for c in 0..n {
            *grad_input.add(base + c) =
                *output.add(base + c) * (*grad_output.add(base + c) - dot);
        }
    }
}
