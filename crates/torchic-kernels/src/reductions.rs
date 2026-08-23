use core::arch::wasm32::*;

// Partial sum for two-phase reduce. Each worker writes into its own slot.
#[no_mangle]
pub unsafe extern "C" fn sum_partial(
    input: *const f32,
    out: *mut f32,
    out_index: u32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    let mut acc = f32x4_splat(0.0);
    let mut i = start;
    while i + 4 <= end {
        let v = v128_load(input.add(i) as *const v128);
        acc = f32x4_add(acc, v);
        i += 4;
    }
    let mut lanes = [0f32; 4];
    v128_store(lanes.as_mut_ptr() as *mut v128, acc);
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while i < end {
        sum += *input.add(i);
        i += 1;
    }
    *out.add(out_index as usize) = sum;
}

#[no_mangle]
pub unsafe extern "C" fn sum_final(input: *const f32, out: *mut f32, n: u32) {
    let mut sum = 0.0f32;
    for i in 0..n as usize {
        sum += *input.add(i);
    }
    *out = sum;
}

// Input is contiguous with shape (outer, axis, inner). Output is (outer, inner).
// out[o, i] = sum over k of input[o, k, i]. Each worker handles a slice of the
// output's flat index range [start, end).
#[no_mangle]
pub unsafe extern "C" fn sum_axis(
    input: *const f32,
    out: *mut f32,
    axis_size: u32,
    inner_size: u32,
    start: u32,
    end: u32,
) {
    let axis_size = axis_size as usize;
    let inner_size = inner_size as usize;
    let start = start as usize;
    let end = end as usize;
    for out_i in start..end {
        let outer = out_i / inner_size;
        let inner = out_i % inner_size;
        let base = outer * axis_size * inner_size + inner;
        let mut sum = 0.0f32;
        for k in 0..axis_size {
            sum += *input.add(base + k * inner_size);
        }
        *out.add(out_i) = sum;
    }
}

// out[i] = a[i] + scalar[0]: broadcast a length-1 tensor across all elements.
#[no_mangle]
pub unsafe extern "C" fn add_scalar_tensor(
    a: *const f32,
    scalar: *const f32,
    out: *mut f32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    let val = *scalar;
    let vv = f32x4_splat(val);
    let mut i = start;
    while i + 4 <= end {
        let av = v128_load(a.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_add(av, vv));
        i += 4;
    }
    while i < end {
        *out.add(i) = *a.add(i) + val;
        i += 1;
    }
}
