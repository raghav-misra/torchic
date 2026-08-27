// Repeat-interleave: duplicate each slice along an axis `repeats` times.
// Treats input as [outer, axis_size, inner] and output as [outer, axis_size * repeats, inner].
// For output index k → (o, d_out, i), map back to input index (o, d_out / repeats, i).
#[no_mangle]
pub unsafe extern "C" fn repeat_interleave(
    input: *const f32,
    output: *mut f32,
    axis_size: u32,
    inner: u32,
    repeats: u32,
    start: u32,
    end: u32,
) {
    let axis_size = axis_size as usize;
    let inner = inner as usize;
    let repeats = repeats as usize;
    let stride_in = axis_size * inner;

    for k in start as usize..end as usize {
        let inner_idx = k % inner;
        let rest = k / inner;
        let d_out = rest % (axis_size * repeats);
        let o = rest / (axis_size * repeats);
        let d_in = d_out / repeats;
        let input_idx = o * stride_in + d_in * inner + inner_idx;
        *output.add(k) = *input.add(input_idx);
    }
}
