// output[i] = weights[indices[i / D] * D + (i % D)] where D = embedding_dim.
#[no_mangle]
pub unsafe extern "C" fn embedding(
    weights: *const f32,
    indices: *const f32,
    output: *mut f32,
    embedding_dim: u32,
    start: u32,
    end: u32,
) {
    let d = embedding_dim as usize;
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let idx = i / d;
        let off = i % d;
        let row = *indices.add(idx) as usize;
        *output.add(i) = *weights.add(row * d + off);
    }
}

// Non-atomic scatter-add. Coordinator must dispatch this on a single worker to avoid races.
#[no_mangle]
pub unsafe extern "C" fn embedding_backward(
    weights_grad: *mut f32,
    indices: *const f32,
    output_grad: *const f32,
    embedding_dim: u32,
    start: u32,
    end: u32,
) {
    let d = embedding_dim as usize;
    let start = start as usize;
    let end = end as usize;
    for i in start..end {
        let idx = i / d;
        let off = i % d;
        let row = *indices.add(idx) as usize;
        *weights_grad.add(row * d + off) += *output_grad.add(i);
    }
}
