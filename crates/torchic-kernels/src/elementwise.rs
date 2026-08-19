use core::arch::wasm32::*;

macro_rules! simd_binary {
    ($name:ident, $op_simd:ident, $op_scalar:tt) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            a: *const f32,
            b: *const f32,
            out: *mut f32,
            start: u32,
            end: u32,
        ) {
            let start = start as usize;
            let end = end as usize;
            let mut i = start;
            while i + 4 <= end {
                let av = v128_load(a.add(i) as *const v128);
                let bv = v128_load(b.add(i) as *const v128);
                v128_store(out.add(i) as *mut v128, $op_simd(av, bv));
                i += 4;
            }
            while i < end {
                *out.add(i) = *a.add(i) $op_scalar *b.add(i);
                i += 1;
            }
        }
    };
}

simd_binary!(add, f32x4_add, +);
simd_binary!(sub, f32x4_sub, -);
simd_binary!(mul, f32x4_mul, *);
simd_binary!(div, f32x4_div, /);

// Broadcast add. out[i] = a[offA(i)] + b[offB(i)], where offX comes from shape/strides.
#[no_mangle]
pub unsafe extern "C" fn add_broadcast(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    start: u32,
    end: u32,
    ndim: u32,
    shape_ptr: *const u32,
    strides_a_ptr: *const u32,
    strides_b_ptr: *const u32,
) {
    let start = start as usize;
    let end = end as usize;
    let ndim = ndim as usize;
    for i in start..end {
        let mut idx = i;
        let mut off_a = 0usize;
        let mut off_b = 0usize;
        let mut d = ndim;
        while d > 0 {
            d -= 1;
            let size = *shape_ptr.add(d) as usize;
            let pos = idx % size;
            idx /= size;
            off_a += pos * (*strides_a_ptr.add(d) as usize);
            off_b += pos * (*strides_b_ptr.add(d) as usize);
        }
        *out.add(i) = *a.add(off_a) + *b.add(off_b);
    }
}

#[no_mangle]
pub unsafe extern "C" fn neg(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    let zero = f32x4_splat(0.0);
    let mut i = start;
    while i + 4 <= end {
        let av = v128_load(a.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_sub(zero, av));
        i += 4;
    }
    while i < end {
        *out.add(i) = -*a.add(i);
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn relu(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    let zero = f32x4_splat(0.0);
    let mut i = start;
    while i + 4 <= end {
        let av = v128_load(a.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_max(av, zero));
        i += 4;
    }
    while i < end {
        let v = *a.add(i);
        *out.add(i) = if v > 0.0 { v } else { 0.0 };
        i += 1;
    }
}

// grad_input[i] = input[i] > 0 ? grad_output[i] : 0
#[no_mangle]
pub unsafe extern "C" fn relu_backward(
    input: *const f32,
    grad_output: *const f32,
    grad_input: *mut f32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        *grad_input.add(i) = if *input.add(i) > 0.0 { *grad_output.add(i) } else { 0.0 };
    }
}

#[no_mangle]
pub unsafe extern "C" fn exp(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        *out.add(i) = libm::expf(*a.add(i));
    }
}

#[no_mangle]
pub unsafe extern "C" fn log(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        *out.add(i) = libm::logf(*a.add(i));
    }
}

#[no_mangle]
pub unsafe extern "C" fn tanh(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        *out.add(i) = libm::tanhf(*a.add(i));
    }
}

// derivative of tanh: grad_input = grad_output * (1 - out^2)
#[no_mangle]
pub unsafe extern "C" fn tanh_backward(
    output: *const f32,
    grad_output: *const f32,
    grad_input: *mut f32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let o = *output.add(i);
        *grad_input.add(i) = *grad_output.add(i) * (1.0 - o * o);
    }
}

#[no_mangle]
pub unsafe extern "C" fn fill(out: *mut f32, val: f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    let vv = f32x4_splat(val);
    let mut i = start;
    while i + 4 <= end {
        v128_store(out.add(i) as *mut v128, vv);
        i += 4;
    }
    while i < end {
        *out.add(i) = val;
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn copy(input: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    let mut i = start;
    while i + 4 <= end {
        let v = v128_load(input.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, v);
        i += 4;
    }
    while i < end {
        *out.add(i) = *input.add(i);
        i += 1;
    }
}

// Gather a strided view into a contiguous output.
#[no_mangle]
pub unsafe extern "C" fn materialize(
    input: *const f32,
    output: *mut f32,
    start: u32,
    end: u32,
    ndim: u32,
    shape_ptr: *const u32,
    strides_ptr: *const u32,
) {
    let start = start as usize;
    let end = end as usize;
    let ndim = ndim as usize;
    for i in start..end {
        let mut idx = i;
        let mut input_offset = 0usize;
        let mut d = ndim;
        while d > 0 {
            d -= 1;
            let size = *shape_ptr.add(d) as usize;
            let pos = idx % size;
            idx /= size;
            input_offset += pos * (*strides_ptr.add(d) as usize);
        }
        *output.add(i) = *input.add(input_offset);
    }
}

// xorshift32 PRNG. Cheap, decent statistical properties for RNG-flavored fill.
#[inline(always)]
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

#[inline(always)]
fn next_unit(state: &mut u32) -> f32 {
    // (0, 1] – avoid 0 so log() in Box-Muller stays finite
    (xorshift32(state) as f32 + 1.0) / 4294967297.0
}

// Standard normal via Box-Muller. Seed is per-call so parallel workers get
// independent streams (JS side derives it from workerIndex + a counter).
#[no_mangle]
pub unsafe extern "C" fn randn(out: *mut f32, start: u32, end: u32, seed: u32) {
    let start = start as usize;
    let end = end as usize;
    let mut state = if seed == 0 { 0x9E3779B9 } else { seed };
    let two_pi = 2.0 * core::f32::consts::PI;
    for i in start..end {
        let u = next_unit(&mut state);
        let v = next_unit(&mut state);
        *out.add(i) = libm::sqrtf(-2.0 * libm::logf(u)) * libm::cosf(two_pi * v);
    }
}
