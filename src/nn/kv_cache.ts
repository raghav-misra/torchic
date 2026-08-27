import { Tensor } from "../frontend/tensor";

// Per-layer streaming KV cache for causal decoder-only inference. Pre-allocates
// `2 * numLayers` tensors of shape [maxSeqLen, numKvHeads, headDim]. Callers
// write new K/V slabs at the current cursor position, then commit(tNew) once
// per token step after all layers have been written.
//
// Layout is [T, H, D]-major so appends are contiguous byte writes at
// `cursor * numKvHeads * headDim` elements into each layer's tensor. Read
// returns a slice view of [0..cursor+tNew, H, D] — caller reshapes/transposes
// to whatever their attention layout expects.
export class KVCache {
  private kCache: Tensor[];
  private vCache: Tensor[];
  private cursor = 0;
  private readonly perTokenElements: number;

  constructor(
    public readonly numLayers: number,
    public readonly maxSeqLen: number,
    public readonly numKvHeads: number,
    public readonly headDim: number,
  ) {
    if (numLayers <= 0 || maxSeqLen <= 0 || numKvHeads <= 0 || headDim <= 0) {
      throw new Error(
        `KVCache: all dims must be positive, got numLayers=${numLayers} maxSeqLen=${maxSeqLen} numKvHeads=${numKvHeads} headDim=${headDim}`,
      );
    }
    this.perTokenElements = numKvHeads * headDim;
    this.kCache = [];
    this.vCache = [];
    for (let i = 0; i < numLayers; i++) {
      this.kCache.push(Tensor.empty([maxSeqLen, numKvHeads, headDim]));
      this.vCache.push(Tensor.empty([maxSeqLen, numKvHeads, headDim]));
    }
  }

  // Append `tNew` tokens' K/V into `layerIdx` at the current cursor and return
  // views of the cache spanning [0..cursor + tNew, numKvHeads, headDim].
  // Cursor is NOT advanced; call commit(tNew) after all layers written.
  // kNew/vNew must contain exactly `tNew * numKvHeads * headDim` contiguous
  // elements each; their shape is otherwise unconstrained.
  write(
    layerIdx: number,
    kNew: Tensor,
    vNew: Tensor,
    tNew: number,
  ): { k: Tensor; v: Tensor } {
    this.validateLayer(layerIdx);
    if (tNew <= 0) throw new Error(`KVCache.write: tNew must be positive, got ${tNew}`);
    if (this.cursor + tNew > this.maxSeqLen) {
      throw new Error(
        `KVCache.write: cursor ${this.cursor} + tNew ${tNew} > maxSeqLen ${this.maxSeqLen}`,
      );
    }
    const expected = tNew * this.perTokenElements;
    const kNumel = kNew.shape.reduce((a, b) => a * b, 1);
    const vNumel = vNew.shape.reduce((a, b) => a * b, 1);
    if (kNumel !== expected || vNumel !== expected) {
      throw new Error(
        `KVCache.write: expected ${expected} elements per new K/V, got kNew=${kNumel} vNew=${vNumel}`,
      );
    }

    const offset = this.cursor * this.perTokenElements;
    this.kCache[layerIdx].copyFrom(kNew, { atOffset: offset });
    this.vCache[layerIdx].copyFrom(vNew, { atOffset: offset });

    const end = this.cursor + tNew;
    const kView = this.kCache[layerIdx].slice([
      [0, end],
      [0, this.numKvHeads],
      [0, this.headDim],
    ]);
    const vView = this.vCache[layerIdx].slice([
      [0, end],
      [0, this.numKvHeads],
      [0, this.headDim],
    ]);
    return { k: kView, v: vView };
  }

  // Advance the cursor after all layers have written for the current step.
  commit(tNew: number): void {
    if (tNew <= 0) throw new Error(`KVCache.commit: tNew must be positive, got ${tNew}`);
    if (this.cursor + tNew > this.maxSeqLen) {
      throw new Error(
        `KVCache.commit: cursor ${this.cursor} + tNew ${tNew} > maxSeqLen ${this.maxSeqLen}`,
      );
    }
    this.cursor += tNew;
  }

  get position(): number {
    return this.cursor;
  }

  reset(): void {
    this.cursor = 0;
  }

  dispose(): void {
    for (const t of this.kCache) t.dispose();
    for (const t of this.vCache) t.dispose();
    this.kCache = [];
    this.vCache = [];
  }

  private validateLayer(layerIdx: number): void {
    if (layerIdx < 0 || layerIdx >= this.numLayers) {
      throw new Error(`KVCache: layer ${layerIdx} out of range [0, ${this.numLayers})`);
    }
  }
}
