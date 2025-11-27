export function embedding(
    weights: Float32Array,
    indices: Float32Array,
    output: Float32Array,
    embeddingDim: number,
    start: number,
    end: number
) {
    // Output is flattened [indices.shape..., embeddingDim]
    // We iterate over the flattened output array
    
    for (let i = start; i < end; i++) {
        // Map flat output index 'i' to:
        // 1. Which index in 'indices' we are using
        // 2. Which dimension of the embedding vector we are copying
        
        const indexInIndices = Math.floor(i / embeddingDim);
        const offsetInEmbedding = i % embeddingDim;
        
        const row = indices[indexInIndices];
        
        // Bounds check (optional but good for safety, though costly in tight loop)
        // Assuming valid indices for now for performance
        
        output[i] = weights[row * embeddingDim + offsetInEmbedding];
    }
}

export function embedding_backward(
    weightsGrad: Float32Array,
    indices: Float32Array,
    outputGrad: Float32Array,
    embeddingDim: number,
    start: number,
    end: number
) {
    // This is tricky to parallelize without atomic adds because multiple indices might point to the same row.
    // However, if we parallelize over the outputGrad (which corresponds to indices), we can just accumulate.
    // BUT, SharedArrayBuffer doesn't support atomic floats easily.
    
    // For now, let's implement a naive version that might have race conditions if parallelized naively 
    // OR we rely on the fact that we might need a different parallelization strategy (e.g. by weight row).
    
    // A common strategy for embedding backward on CPU is to just iterate and add.
    // Since we are in a worker, we are writing to 'weightsGrad' which is shared.
    // If multiple workers write to the same weight row, we have a race.
    
    // For this simplified engine, let's assume single-threaded backward for embeddings 
    // OR use Atomics if we can (but Atomics only work on Int32).
    
    // ALTERNATIVE: Each worker computes a partial gradient for weights, and then we sum them.
    // That's expensive memory-wise.
    
    // Given the constraints and the "toy" nature of this library, 
    // maybe we can just use a lock or accept the race condition (bad) 
    // or just run this op on a single worker (Coordinator logic).
    
    // Let's implement the kernel assuming it might be run on one worker for safety, 
    // or we accept we need a "scatter_add" primitive.
    
    // Let's just write the loop.
    for (let i = start; i < end; i++) {
        const indexInIndices = Math.floor(i / embeddingDim);
        const offsetInEmbedding = i % embeddingDim;
        
        const row = indices[indexInIndices];
        const grad = outputGrad[i];
        
        // weightsGrad[row * embeddingDim + offsetInEmbedding] += grad;
        // This += is not atomic.
        
        // We can't easily fix this without atomic float support or architectural changes.
        // For now, I will implement it, but we should probably run this on 1 worker.
        
        weightsGrad[row * embeddingDim + offsetInEmbedding] += grad;
    }
}
