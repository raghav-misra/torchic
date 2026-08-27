// Row-parallel RoPE (half-split, HF Llama convention).
// Input is a [total_rows, D] view of a [..., T, D] tensor where each row r
// corresponds to time step `r % t_seq` and cos/sin tables have shape [t_seq, D/2].
//   x'[i]         = x[i] * cos[t, i] - x[i + D/2] * sin[t, i]
//   x'[i + D/2]   = x[i] * sin[t, i] + x[i + D/2] * cos[t, i]
#[no_mangle]
pub unsafe extern "C" fn rope(
    x: *const f32,
    cos: *const f32,
    sin: *const f32,
    out: *mut f32,
    t_seq: u32,
    d_half: u32,
    start_row: u32,
    end_row: u32,
) {
    let t_seq = t_seq as usize;
    let d_half = d_half as usize;
    let d = 2 * d_half;
    let start_row = start_row as usize;
    let end_row = end_row as usize;

    for r in start_row..end_row {
        let time = r % t_seq;
        let x_base = r * d;
        let cs_base = time * d_half;
        for i in 0..d_half {
            let a = *x.add(x_base + i);
            let b = *x.add(x_base + i + d_half);
            let c = *cos.add(cs_base + i);
            let s = *sin.add(cs_base + i);
            *out.add(x_base + i) = a * c - b * s;
            *out.add(x_base + i + d_half) = a * s + b * c;
        }
    }
}
