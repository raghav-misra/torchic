// Slice-write primitive: copy count f32 elements from src[0..count] into
// dst[dst_offset..dst_offset + count]. Other positions in the destination
// are left untouched. Foundation for KV cache append.

struct CopyRangeU {
  src: u32,
  dst: u32,
  dst_offset: u32,
  count: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: CopyRangeU;

@compute @workgroup_size(256)
fn copy_range(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  heap[u.dst + u.dst_offset + i] = heap[u.src + i];
}
