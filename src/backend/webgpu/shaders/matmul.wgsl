// Tiled matmul with cooperative workgroup-shared loads.
// Each workgroup covers a 16x16 tile of the output. The K dimension is walked
// in blocks of 16: on each iteration all 256 threads cooperatively load a
// 16x16 tile of A and B into workgroup shared memory, barrier, then do 16
// FMAs each from shared memory.
//
// A register-blocked variant (each thread producing a 4x4 output block) was
// tried and lost to this version. 16-way bank conflicts on shared reads
// outweighed the theoretical arithmetic-intensity win. Getting past this
// version on Turing needs vec4 loads + transposed thread mapping + 128x128
// tiles, which is significant shader engineering.

struct Uniforms {
  a_off: u32,
  b_off: u32,
  out_off: u32,
  m: u32,
  n: u32,
  k: u32,
  start_row: u32,
  end_row: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: Uniforms;

const TILE: u32 = 16u;

var<workgroup> a_tile: array<array<f32, 16>, 16>;
var<workgroup> b_tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = u.start_row + gid.x;
  let col = gid.y;
  let lx = lid.x;
  let ly = lid.y;

  var sum: f32 = 0.0;

  var k_block: u32 = 0u;
  loop {
    if (k_block >= u.k) { break; }

    let a_col = k_block + ly;
    if (row < u.end_row && a_col < u.k) {
      a_tile[lx][ly] = heap[u.a_off + row * u.k + a_col];
    } else {
      a_tile[lx][ly] = 0.0;
    }

    let b_row = k_block + lx;
    if (b_row < u.k && col < u.n) {
      b_tile[lx][ly] = heap[u.b_off + b_row * u.n + col];
    } else {
      b_tile[lx][ly] = 0.0;
    }

    workgroupBarrier();

    for (var p: u32 = 0u; p < TILE; p = p + 1u) {
      sum = sum + a_tile[lx][p] * b_tile[p][ly];
    }

    workgroupBarrier();

    k_block = k_block + TILE;
  }

  if (row < u.end_row && col < u.n) {
    heap[u.out_off + row * u.n + col] = sum;
  }
}
