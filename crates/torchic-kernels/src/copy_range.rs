// Slice-write primitive: copy f32 elements from `src[start..end]` into
// `dst[dst_offset + start..dst_offset + end]`. Other positions in the
// destination are left untouched. Foundation for KV cache append.
#[no_mangle]
pub unsafe extern "C" fn copy_range(
    src: *const f32,
    dst: *mut f32,
    dst_offset: u32,
    start: u32,
    end: u32,
) {
    let dst_offset = dst_offset as usize;
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        *dst.add(dst_offset + i) = *src.add(i);
    }
}
