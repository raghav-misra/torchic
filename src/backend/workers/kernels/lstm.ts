// Fused LSTM inference step (workers backend). Mirrors
// src/backend/webgpu/shaders/lstm_step.wgsl. One call per timestep per
// direction, replacing the ~20 sequential dispatches of the composed path.

export function lstm_step(
  x: Float32Array,
  h: Float32Array,
  c: Float32Array,
  wIh: Float32Array,
  wHh: Float32Array,
  bIh: Float32Array,
  bHh: Float32Array,
  hOut: Float32Array,
  hOutOffset: number,
  cOut: Float32Array,
  cOutOffset: number,
  batchSize: number,
  hidden: number,
  inSize: number,
): void {
  const H = hidden;
  const IN = inSize;
  for (let b = 0; b < batchSize; b++) {
    const xBase = b * IN;
    const hBase = b * H;
    const cBase = b * H;
    const hOutBase = hOutOffset + b * H;
    const cOutBase = cOutOffset + b * H;

    for (let k = 0; k < H; k++) {
      let preI = bIh[0 * H + k] + bHh[0 * H + k];
      let preF = bIh[1 * H + k] + bHh[1 * H + k];
      let preG = bIh[2 * H + k] + bHh[2 * H + k];
      let preO = bIh[3 * H + k] + bHh[3 * H + k];

      const rowI_ih = (0 * H + k) * IN;
      const rowF_ih = (1 * H + k) * IN;
      const rowG_ih = (2 * H + k) * IN;
      const rowO_ih = (3 * H + k) * IN;
      for (let i = 0; i < IN; i++) {
        const xi = x[xBase + i];
        preI += xi * wIh[rowI_ih + i];
        preF += xi * wIh[rowF_ih + i];
        preG += xi * wIh[rowG_ih + i];
        preO += xi * wIh[rowO_ih + i];
      }

      const rowI_hh = (0 * H + k) * H;
      const rowF_hh = (1 * H + k) * H;
      const rowG_hh = (2 * H + k) * H;
      const rowO_hh = (3 * H + k) * H;
      for (let j = 0; j < H; j++) {
        const hj = h[hBase + j];
        preI += hj * wHh[rowI_hh + j];
        preF += hj * wHh[rowF_hh + j];
        preG += hj * wHh[rowG_hh + j];
        preO += hj * wHh[rowO_hh + j];
      }

      const ig = 1 / (1 + Math.exp(-preI));
      const fg = 1 / (1 + Math.exp(-preF));
      const gc = Math.tanh(preG);
      const og = 1 / (1 + Math.exp(-preO));

      const cPrev = c[cBase + k];
      const cNew = fg * cPrev + ig * gc;
      const hNew = og * Math.tanh(cNew);

      hOut[hOutBase + k] = hNew;
      cOut[cOutBase + k] = cNew;
    }
  }
}
