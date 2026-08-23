// 1-D convolution kernels. Forward-only (Kokoro serving is inference).
// Input layout matches PyTorch: input [B, C_in, L_in], weight [C_out, C_in/G, K],
// output [B, C_out, L_out]. G = groups: G=1 standard, G=C_in=C_out depthwise.

export function conv1d(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array | null,
  out: Float32Array,
  B: number,
  Cin: number,
  Lin: number,
  Cout: number,
  K: number,
  Lout: number,
  stride: number,
  pad: number,
  dil: number,
  groups: number,
  startBatch: number,
  endBatch: number,
) {
  const cinPerG = Cin / groups;
  const coutPerG = Cout / groups;
  const inStrideB = Cin * Lin;
  const inStrideC = Lin;
  const wStrideCo = cinPerG * K;
  const wStrideCi = K;
  const outStrideB = Cout * Lout;
  const outStrideC = Lout;

  for (let b = startBatch; b < endBatch; b++) {
    const inB = b * inStrideB;
    const outB = b * outStrideB;
    for (let co = 0; co < Cout; co++) {
      const g = Math.floor(co / coutPerG);
      const ciStart = g * cinPerG;
      const outBase = outB + co * outStrideC;
      const wCo = co * wStrideCo;
      const biasVal = bias ? bias[co] : 0;
      for (let lo = 0; lo < Lout; lo++) {
        let sum = biasVal;
        for (let ciOff = 0; ciOff < cinPerG; ciOff++) {
          const ci = ciStart + ciOff;
          const inC = inB + ci * inStrideC;
          const wCi = wCo + ciOff * wStrideCi;
          for (let k = 0; k < K; k++) {
            const li = lo * stride + k * dil - pad;
            if (li >= 0 && li < Lin) {
              sum += input[inC + li] * weight[wCi + k];
            }
          }
        }
        out[outBase + lo] = sum;
      }
    }
  }
}

// Weight layout matches PyTorch: [C_in, C_out/G, K].
export function conv_transpose1d(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array | null,
  out: Float32Array,
  B: number,
  Cin: number,
  Lin: number,
  Cout: number,
  K: number,
  Lout: number,
  stride: number,
  pad: number,
  dil: number,
  groups: number,
  startBatch: number,
  endBatch: number,
) {
  const cinPerG = Cin / groups;
  const coutPerG = Cout / groups;
  const inStrideB = Cin * Lin;
  const inStrideC = Lin;
  const wStrideCi = coutPerG * K;
  const wStrideCo = K;
  const outStrideB = Cout * Lout;
  const outStrideC = Lout;

  for (let b = startBatch; b < endBatch; b++) {
    const inB = b * inStrideB;
    const outB = b * outStrideB;
    for (let co = 0; co < Cout; co++) {
      const outBase = outB + co * outStrideC;
      const biasVal = bias ? bias[co] : 0;
      for (let lo = 0; lo < Lout; lo++) out[outBase + lo] = biasVal;
    }
    for (let ci = 0; ci < Cin; ci++) {
      const g = Math.floor(ci / cinPerG);
      const coStart = g * coutPerG;
      const inC = inB + ci * inStrideC;
      const wCi = ci * wStrideCi;
      for (let li = 0; li < Lin; li++) {
        const xv = input[inC + li];
        if (xv === 0) continue;
        for (let coOff = 0; coOff < coutPerG; coOff++) {
          const co = coStart + coOff;
          const outBase = outB + co * outStrideC;
          const wBase = wCi + coOff * wStrideCo;
          for (let k = 0; k < K; k++) {
            const lo = li * stride + k * dil - pad;
            if (lo >= 0 && lo < Lout) {
              out[outBase + lo] += xv * weight[wBase + k];
            }
          }
        }
      }
    }
  }
}

export function conv1dOutputLen(
  Lin: number,
  K: number,
  stride: number,
  pad: number,
  dil: number,
): number {
  return Math.floor((Lin + 2 * pad - dil * (K - 1) - 1) / stride) + 1;
}

export function convTranspose1dOutputLen(
  Lin: number,
  K: number,
  stride: number,
  pad: number,
  dil: number,
  outputPad: number,
): number {
  return (Lin - 1) * stride - 2 * pad + dil * (K - 1) + outputPad + 1;
}
