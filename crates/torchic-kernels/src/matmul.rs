use core::arch::wasm32::*;

const MR: usize = 4;
const NR: usize = 8;
const KC: usize = 256;

// Pack A[i:i+MR, pc:pc+kc] into ap laid out as Ap[p*MR + r].
// The MR values for a given k are now adjacent in memory so the microkernel
// can grab them with a single 128-bit access per row-splat.
#[inline]
unsafe fn pack_a(
    a: *const f32,
    ap: *mut f32,
    i: usize,
    pc: usize,
    kc: usize,
    a_rs: usize,
    a_cs: usize,
) {
    for p in 0..kc {
        let dst = ap.add(p * MR);
        *dst.add(0) = *a.add(i * a_rs + (pc + p) * a_cs);
        *dst.add(1) = *a.add((i + 1) * a_rs + (pc + p) * a_cs);
        *dst.add(2) = *a.add((i + 2) * a_rs + (pc + p) * a_cs);
        *dst.add(3) = *a.add((i + 3) * a_rs + (pc + p) * a_cs);
    }
}

// 4x8 register microkernel driven by a packed A panel.
// If `first == true`, accumulators start at zero (fresh tile).
// Otherwise they load prior partial sums from C, so we can accumulate across
// K-blocks without rounding drift beyond a single-pass reduction.
#[inline(always)]
unsafe fn microkernel_4x8(
    ap: *const f32,
    bp: *const f32,
    cp: *mut f32,
    kc: usize,
    b_rs: usize,
    n: usize,
    first: bool,
) {
    let mut c00;
    let mut c01;
    let mut c10;
    let mut c11;
    let mut c20;
    let mut c21;
    let mut c30;
    let mut c31;

    if first {
        let z = f32x4_splat(0.0);
        c00 = z;
        c01 = z;
        c10 = z;
        c11 = z;
        c20 = z;
        c21 = z;
        c30 = z;
        c31 = z;
    } else {
        c00 = v128_load(cp as *const v128);
        c01 = v128_load(cp.add(4) as *const v128);
        c10 = v128_load(cp.add(n) as *const v128);
        c11 = v128_load(cp.add(n + 4) as *const v128);
        c20 = v128_load(cp.add(2 * n) as *const v128);
        c21 = v128_load(cp.add(2 * n + 4) as *const v128);
        c30 = v128_load(cp.add(3 * n) as *const v128);
        c31 = v128_load(cp.add(3 * n + 4) as *const v128);
    }

    for p in 0..kc {
        let a_base = ap.add(p * MR);
        let a0 = v128_load32_splat(a_base as *const u32);
        let a1 = v128_load32_splat(a_base.add(1) as *const u32);
        let a2 = v128_load32_splat(a_base.add(2) as *const u32);
        let a3 = v128_load32_splat(a_base.add(3) as *const u32);

        let b_row = bp.add(p * b_rs);
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

    v128_store(cp as *mut v128, c00);
    v128_store(cp.add(4) as *mut v128, c01);
    v128_store(cp.add(n) as *mut v128, c10);
    v128_store(cp.add(n + 4) as *mut v128, c11);
    v128_store(cp.add(2 * n) as *mut v128, c20);
    v128_store(cp.add(2 * n + 4) as *mut v128, c21);
    v128_store(cp.add(3 * n) as *mut v128, c30);
    v128_store(cp.add(3 * n + 4) as *mut v128, c31);
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

// C[start_row..end_row, :] = A[start_row..end_row, :] * B[:, :]
//
// Fast path: contiguous B (b_col_stride == 1) and n divisible by NR.
// SIMD path packs an MR x KC panel of A per K-block so the microkernel
// hits fully-contiguous, cache-hot A. Non-SIMD B and right/bottom edges
// fall back to a scalar cleanup that preserves arbitrary strides.
#[no_mangle]
//
// scratch: caller-provided scratch region of at least MR*KC f32s, private to
// this call (each worker gets its own slice). We can't stack-alloc it because
// every worker instance's wasm __stack_pointer starts at the same value, so
// they'd race on the same region of shared linear memory.
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
    scratch: *mut f32,
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
    let full_j_end = if can_simd { (n / NR) * NR } else { 0 };

    let a_panel_ptr = scratch;

    let mut i = start_row;
    while i + MR <= end_row {
        if full_j_end > 0 {
            let mut pc = 0usize;
            while pc < k {
                let kc = core::cmp::min(KC, k - pc);
                pack_a(a, a_panel_ptr, i, pc, kc, a_rs, a_cs);

                let mut j = 0usize;
                while j < full_j_end {
                    microkernel_4x8(
                        a_panel_ptr,
                        b.add(pc * b_rs + j),
                        out.add(i * n + j),
                        kc,
                        b_rs,
                        n,
                        pc == 0,
                    );
                    j += NR;
                }
                pc += KC;
            }
        }
        if full_j_end < n {
            scalar_tile(
                a, b, out, i, i + MR, full_j_end, n, k, n, a_rs, a_cs, b_rs, b_cs,
            );
        }
        i += MR;
    }
    if i < end_row {
        scalar_tile(a, b, out, i, end_row, 0, n, k, n, a_rs, a_cs, b_rs, b_cs);
    }
}