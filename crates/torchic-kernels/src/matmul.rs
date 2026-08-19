use core::arch::wasm32::*;

// C = A * B, C[start_row..end_row, :]
// Blocked matmul; SIMD fast-path when B is column-contiguous (b_col_stride == 1).
#[no_mangle]
pub unsafe extern "C" fn matmul(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: u32,
    n: u32,
    k: u32,
    start_row: u32,
    end_row: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
) {
    let _ = m;
    let n = n as usize;
    let k = k as usize;
    let start_row = start_row as usize;
    let end_row = end_row as usize;
    let a_rs = a_row_stride as usize;
    let a_cs = a_col_stride as usize;
    let b_rs = b_row_stride as usize;
    let b_cs = b_col_stride as usize;

    const BLOCK: usize = 32;
    let b_contig = b_cs == 1;

    let mut i0 = start_row;
    while i0 < end_row {
        let i_max = core::cmp::min(i0 + BLOCK, end_row);
        let mut j0 = 0usize;
        while j0 < n {
            let j_max = core::cmp::min(j0 + BLOCK, n);
            let mut p0 = 0usize;
            while p0 < k {
                let p_max = core::cmp::min(p0 + BLOCK, k);
                for i in i0..i_max {
                    let a_row_base = i * a_rs;
                    let out_row_base = i * n;
                    if p0 == 0 {
                        for j in j0..j_max {
                            *out.add(out_row_base + j) = 0.0;
                        }
                    }
                    for p in p0..p_max {
                        let a_val = *a.add(a_row_base + p * a_cs);
                        let b_row_base = p * b_rs;
                        if b_contig {
                            let a_splat = f32x4_splat(a_val);
                            let mut j = j0;
                            while j + 4 <= j_max {
                                let out_ptr = out.add(out_row_base + j) as *mut v128;
                                let b_ptr = b.add(b_row_base + j) as *const v128;
                                let acc = v128_load(out_ptr as *const v128);
                                let bv = v128_load(b_ptr);
                                v128_store(out_ptr, f32x4_add(acc, f32x4_mul(a_splat, bv)));
                                j += 4;
                            }
                            while j < j_max {
                                *out.add(out_row_base + j) +=
                                    a_val * *b.add(b_row_base + j);
                                j += 1;
                            }
                        } else {
                            for j in j0..j_max {
                                *out.add(out_row_base + j) +=
                                    a_val * *b.add(b_row_base + j * b_cs);
                            }
                        }
                    }
                }
                p0 += BLOCK;
            }
            j0 += BLOCK;
        }
        i0 += BLOCK;
    }
}
