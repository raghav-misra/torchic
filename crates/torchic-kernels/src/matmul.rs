use core::arch::wasm32::*;

// 4x8 register-blocked microkernel. Eight f32x4 accumulators live in wasm
// locals (→ CPU SIMD registers), amortizing load/store cost across k and
// exposing 8 independent FMA chains for the CPU to pipeline.
#[inline(always)]
unsafe fn microkernel_4x8(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    k: usize,
    a_rs: usize,
    a_cs: usize,
    b_rs: usize,
    n: usize,
) {
    let mut c00 = f32x4_splat(0.0);
    let mut c01 = f32x4_splat(0.0);
    let mut c10 = f32x4_splat(0.0);
    let mut c11 = f32x4_splat(0.0);
    let mut c20 = f32x4_splat(0.0);
    let mut c21 = f32x4_splat(0.0);
    let mut c30 = f32x4_splat(0.0);
    let mut c31 = f32x4_splat(0.0);

    for p in 0..k {
        let a0 = f32x4_splat(*a_ptr.add(p * a_cs));
        let a1 = f32x4_splat(*a_ptr.add(a_rs + p * a_cs));
        let a2 = f32x4_splat(*a_ptr.add(2 * a_rs + p * a_cs));
        let a3 = f32x4_splat(*a_ptr.add(3 * a_rs + p * a_cs));

        let b_row = b_ptr.add(p * b_rs);
        let b0 = v128_load(b_row as *const v128);
        let b1 = v128_load(b_row.add(4) as *const v128);

        c00 = f32x4_add(c00, f32x4_mul(a0, b0));
        c01 = f32x4_add(c01, f32x4_mul(a0, b1));
        c10 = f32x4_add(c10, f32x4_mul(a1, b0));
        c11 = f32x4_add(c11, f32x4_mul(a1, b1));
        c20 = f32x4_add(c20, f32x4_mul(a2, b0));
        c21 = f32x4_add(c21, f32x4_mul(a2, b1));
        c30 = f32x4_add(c30, f32x4_mul(a3, b0));
        c31 = f32x4_add(c31, f32x4_mul(a3, b1));
    }

    v128_store(c_ptr as *mut v128, c00);
    v128_store(c_ptr.add(4) as *mut v128, c01);
    v128_store(c_ptr.add(n) as *mut v128, c10);
    v128_store(c_ptr.add(n + 4) as *mut v128, c11);
    v128_store(c_ptr.add(2 * n) as *mut v128, c20);
    v128_store(c_ptr.add(2 * n + 4) as *mut v128, c21);
    v128_store(c_ptr.add(3 * n) as *mut v128, c30);
    v128_store(c_ptr.add(3 * n + 4) as *mut v128, c31);
}

#[inline]
unsafe fn scalar_tile(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    k: usize,
    n: usize,
    a_rs: usize,
    a_cs: usize,
    b_rs: usize,
    b_cs: usize,
) {
    for i in i_start..i_end {
        for j in j_start..j_end {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *a.add(i * a_rs + p * a_cs) * *b.add(p * b_rs + j * b_cs);
            }
            *out.add(i * n + j) = sum;
        }
    }
}

const MR: usize = 4;
const NR: usize = 8;

// C[start_row..end_row, :] = A[start_row..end_row, :] * B[:, :]
// SIMD fast path requires b_col_stride == 1. Full arbitrary-stride support
// via the scalar fallback for edge tiles and non-contiguous B.
#[no_mangle]
pub unsafe extern "C" fn matmul(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    _m: u32,
    n: u32,
    k: u32,
    start_row: u32,
    end_row: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
) {
    let n = n as usize;
    let k = k as usize;
    let start_row = start_row as usize;
    let end_row = end_row as usize;
    let a_rs = a_row_stride as usize;
    let a_cs = a_col_stride as usize;
    let b_rs = b_row_stride as usize;
    let b_cs = b_col_stride as usize;

    let can_simd = b_cs == 1;

    let mut i = start_row;
    while i + MR <= end_row {
        let mut j = 0usize;
        if can_simd {
            while j + NR <= n {
                microkernel_4x8(
                    a.add(i * a_rs),
                    b.add(j),
                    out.add(i * n + j),
                    k,
                    a_rs,
                    a_cs,
                    b_rs,
                    n,
                );
                j += NR;
            }
        }
        if j < n {
            scalar_tile(a, b, out, i, i + MR, j, n, k, n, a_rs, a_cs, b_rs, b_cs);
        }
        i += MR;
    }
    if i < end_row {
        scalar_tile(a, b, out, i, end_row, 0, n, k, n, a_rs, a_cs, b_rs, b_cs);
    }
}
