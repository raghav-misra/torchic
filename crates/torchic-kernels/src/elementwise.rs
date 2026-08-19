use core::arch::wasm32::*;

// Contiguous add. out[i] = a[i] + b[i], i in start..end.
#[no_mangle]
pub unsafe extern "C" fn add(
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
        v128_store(out.add(i) as *mut v128, f32x4_add(av, bv));
        i += 4;
    }
    while i < end {
        *out.add(i) = *a.add(i) + *b.add(i);
        i += 1;
    }
}

// Broadcast add. out[i] = a[offA(i)] + b[offB(i)], where offX comes from shape/strides.
// shape/strides passed as raw pointers into WASM memory (u32 arrays), rank = ndim.
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
    // (0, 1] — avoid 0 so log() in Box-Muller stays finite
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
