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

#[no_mangle]
pub unsafe extern "C" fn sin(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        *out.add(i) = libm::sinf(*a.add(i));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cos(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        *out.add(i) = libm::cosf(*a.add(i));
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

// Tanh approximation used by BERT / GPT-2 / Kokoro.
// gelu(x) = 0.5 * x * (1 + tanh(c * (x + b * x^3)))
const GELU_C: f32 = 0.7978845608028654;
const GELU_B: f32 = 0.044715;

#[no_mangle]
pub unsafe extern "C" fn gelu(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let x = *a.add(i);
        let u = GELU_C * (x + GELU_B * x * x * x);
        *out.add(i) = 0.5 * x * (1.0 + libm::tanhf(u));
    }
}

#[no_mangle]
pub unsafe extern "C" fn gelu_backward(
    input: *const f32,
    grad_output: *const f32,
    grad_input: *mut f32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let x = *input.add(i);
        let x2 = x * x;
        let u = GELU_C * (x + GELU_B * x * x2);
        let t = libm::tanhf(u);
        let dudx = GELU_C * (1.0 + 3.0 * GELU_B * x2);
        let dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dudx;
        *grad_input.add(i) = *grad_output.add(i) * dgelu;
    }
}

#[no_mangle]
pub unsafe extern "C" fn sqrt_op(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    let mut i = start;
    // f32x4_sqrt maps to WASM's native f32x4.sqrt (hardware sqrt), unlike
    // libm::sqrtf which is a software polynomial approximation.
    while i + 4 <= end {
        let av = v128_load(a.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_sqrt(av));
        i += 4;
    }
    while i < end {
        let scratch: [f32; 4] = [*a.add(i), 0.0, 0.0, 0.0];
        let v = v128_load(scratch.as_ptr() as *const v128);
        let r = f32x4_sqrt(v);
        let mut dst: [f32; 4] = [0.0; 4];
        v128_store(dst.as_mut_ptr() as *mut v128, r);
        *out.add(i) = dst[0];
        i += 1;
    }
}

// d/dx sqrt(x) = 0.5 / sqrt(x) = 0.5 / y
#[no_mangle]
pub unsafe extern "C" fn sqrt_backward(
    output: *const f32,
    grad_output: *const f32,
    grad_input: *mut f32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        *grad_input.add(i) = *grad_output.add(i) * 0.5 / *output.add(i);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rsqrt_op(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    let one = f32x4_splat(1.0);
    let mut i = start;
    while i + 4 <= end {
        let av = v128_load(a.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_div(one, f32x4_sqrt(av)));
        i += 4;
    }
    while i < end {
        let scratch: [f32; 4] = [*a.add(i), 1.0, 1.0, 1.0];
        let v = v128_load(scratch.as_ptr() as *const v128);
        let r = f32x4_div(one, f32x4_sqrt(v));
        let mut dst: [f32; 4] = [0.0; 4];
        v128_store(dst.as_mut_ptr() as *mut v128, r);
        *out.add(i) = dst[0];
        i += 1;
    }
}

// d/dx x^(-1/2) = -0.5 * x^(-3/2) = -0.5 * y^3
#[no_mangle]
pub unsafe extern "C" fn rsqrt_backward(
    output: *const f32,
    grad_output: *const f32,
    grad_input: *mut f32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let y = *output.add(i);
        *grad_input.add(i) = *grad_output.add(i) * -0.5 * y * y * y;
    }
}

#[no_mangle]
pub unsafe extern "C" fn sigmoid(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let x = *a.add(i);
        *out.add(i) = 1.0 / (1.0 + libm::expf(-x));
    }
}

// d/dx sigmoid(x) = y * (1 - y)
#[no_mangle]
pub unsafe extern "C" fn sigmoid_backward(
    output: *const f32,
    grad_output: *const f32,
    grad_input: *mut f32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let y = *output.add(i);
        *grad_input.add(i) = *grad_output.add(i) * y * (1.0 - y);
    }
}

#[no_mangle]
pub unsafe extern "C" fn leaky_relu(
    a: *const f32,
    out: *mut f32,
    negative_slope: f32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let x = *a.add(i);
        *out.add(i) = if x >= 0.0 { x } else { x * negative_slope };
    }
}

#[no_mangle]
pub unsafe extern "C" fn leaky_relu_backward(
    input: *const f32,
    grad_output: *const f32,
    grad_input: *mut f32,
    negative_slope: f32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let x = *input.add(i);
        let s = if x >= 0.0 { 1.0 } else { negative_slope };
        *grad_input.add(i) = *grad_output.add(i) * s;
    }
}

// SiLU / Swish: y = x * sigmoid(x)
#[no_mangle]
pub unsafe extern "C" fn silu(a: *const f32, out: *mut f32, start: u32, end: u32) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let x = *a.add(i);
        let s = 1.0 / (1.0 + libm::expf(-x));
        *out.add(i) = x * s;
    }
}

// d/dx (x*sigmoid(x)) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
#[no_mangle]
pub unsafe extern "C" fn silu_backward(
    input: *const f32,
    grad_output: *const f32,
    grad_input: *mut f32,
    start: u32,
    end: u32,
) {
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let x = *input.add(i);
        let s = 1.0 / (1.0 + libm::expf(-x));
        *grad_input.add(i) = *grad_output.add(i) * s * (1.0 + x * (1.0 - s));
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
    // (0, 1]: avoid 0 so log() in Box-Muller stays finite
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
