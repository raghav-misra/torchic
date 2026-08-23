// Concat: copy a contiguous input tensor into a strided slab of the output.
// Meant to be called N times, once per concat operand, with the appropriate
// axis_offset. The kernel is per-input-element; it maps each input index into
// its (outer, in_axis, inner) coordinates and writes at (outer, axis_offset +
// in_axis, inner) in the output layout.

#[no_mangle]
pub unsafe extern "C" fn concat_slab(
    input: *const f32,
    out: *mut f32,
    outer_size: u32,
    in_axis_size: u32,
    out_axis_size: u32,
    axis_offset: u32,
    inner_size: u32,
    start: u32,
    end: u32,
) {
    let in_axis_size = in_axis_size as usize;
    let out_axis_size = out_axis_size as usize;
    let axis_offset = axis_offset as usize;
    let inner_size = inner_size as usize;
    let start = start as usize;
    let end = end as usize;

    let _total = (outer_size as usize) * in_axis_size * inner_size;
    let axis_x_inner = in_axis_size * inner_size;
    let out_axis_x_inner = out_axis_size * inner_size;

    for idx in start..end {
        let outer = idx / axis_x_inner;
        let rem = idx % axis_x_inner;
        let axis = rem / inner_size;
        let inner = rem % inner_size;
        let out_idx = outer * out_axis_x_inner + (axis_offset + axis) * inner_size + inner;
        *out.add(out_idx) = *input.add(idx);
    }
}
