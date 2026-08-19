// output[r, c] = input[c, r] for r in [start_row, end_row), c in [0, m).
// input shape: [m, n], output shape: [n, m].
#[no_mangle]
pub unsafe extern "C" fn transpose(
    input: *const f32,
    output: *mut f32,
    m: u32,
    n: u32,
    start_row: u32,
    end_row: u32,
) {
    let m = m as usize;
    let n = n as usize;
    let start_row = start_row as usize;
    let end_row = end_row as usize;
    for r in start_row..end_row {
        for c in 0..m {
            *output.add(r * m + c) = *input.add(c * n + r);
        }
    }
}
