// STFT / ISTFT for Kokoro's ISTFT-based vocoder tail. Main-thread f32; no
// backend involvement. FFT length must be a power of two.

function bitReverse(x: number, bits: number): number {
  let r = 0;
  for (let i = 0; i < bits; i++) {
    r = (r << 1) | (x & 1);
    x >>>= 1;
  }
  return r;
}

// Iterative radix-2 Cooley-Tukey when n is a power of 2; direct O(n²) DFT
// otherwise. In-place on both real and imag. The direct path is used by
// Kokoro's ISTFT tail (n_fft=20).
export function fft(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if (n !== imag.length) throw new Error(`fft: length mismatch ${n} vs ${imag.length}`);
  if ((n & (n - 1)) !== 0) {
    dftDirect(real, imag, /*forward*/ true);
    return;
  }
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    if (j > i) {
      let tmp = real[i];
      real[i] = real[j];
      real[j] = tmp;
      tmp = imag[i];
      imag[i] = imag[j];
      imag[j] = tmp;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = -2 * Math.PI / size;
    for (let i = 0; i < n; i += size) {
      for (let k = 0; k < half; k++) {
        const angle = step * k;
        const wr = Math.cos(angle);
        const wi = Math.sin(angle);
        const jr = real[i + k + half];
        const ji = imag[i + k + half];
        const tr = wr * jr - wi * ji;
        const ti = wr * ji + wi * jr;
        const ur = real[i + k];
        const ui = imag[i + k];
        real[i + k] = ur + tr;
        imag[i + k] = ui + ti;
        real[i + k + half] = ur - tr;
        imag[i + k + half] = ui - ti;
      }
    }
  }
}

// O(n²) DFT/iDFT. Only used when n is not a power of 2; the twiddle table is
// computed once per call, which is fine for the small sizes (n_fft=20) that
// need this path.
function dftDirect(real: Float32Array, imag: Float32Array, forward: boolean): void {
  const n = real.length;
  const sign = forward ? -1 : 1;
  const outR = new Float32Array(n);
  const outI = new Float32Array(n);
  const base = (sign * 2 * Math.PI) / n;
  for (let k = 0; k < n; k++) {
    let sumR = 0;
    let sumI = 0;
    for (let j = 0; j < n; j++) {
      const angle = base * k * j;
      const wr = Math.cos(angle);
      const wi = Math.sin(angle);
      sumR += real[j] * wr - imag[j] * wi;
      sumI += real[j] * wi + imag[j] * wr;
    }
    outR[k] = sumR;
    outI[k] = sumI;
  }
  real.set(outR);
  imag.set(outI);
}

export function ifft(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if ((n & (n - 1)) !== 0) {
    dftDirect(real, imag, /*forward*/ false);
    const inv = 1 / n;
    for (let i = 0; i < n; i++) {
      real[i] *= inv;
      imag[i] *= inv;
    }
    return;
  }
  for (let i = 0; i < n; i++) imag[i] = -imag[i];
  fft(real, imag);
  const inv = 1 / n;
  for (let i = 0; i < n; i++) {
    real[i] *= inv;
    imag[i] = -imag[i] * inv;
  }
}

export function hannWindow(n: number, periodic = true): Float32Array {
  const w = new Float32Array(n);
  const denom = periodic ? n : n - 1;
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / denom);
  return w;
}

export interface STFTOptions {
  nFFT: number;
  hopLength: number;
  winLength?: number; // defaults to nFFT
  window?: Float32Array; // defaults to Hann of size winLength, zero-padded to nFFT
  center?: boolean; // if true, pads by nFFT/2 at both ends (librosa default)
}

export interface STFTResult {
  real: Float32Array; // shape [numBins, numFrames]
  imag: Float32Array;
  numBins: number;
  numFrames: number;
}

// STFT of a real-valued signal. Returns the one-sided spectrum
// (bins 0..nFFT/2 inclusive) in row-major [numBins, numFrames] layout.
export function stft(signal: Float32Array, opts: STFTOptions): STFTResult {
  const { nFFT, hopLength } = opts;
  const winLength = opts.winLength ?? nFFT;
  const center = opts.center ?? true;
  if (winLength > nFFT) throw new Error(`stft: winLength ${winLength} > nFFT ${nFFT}`);

  const window = padWindow(opts.window ?? hannWindow(winLength), nFFT);
  const padded = center ? padReflect(signal, nFFT >> 1) : signal;
  const numBins = (nFFT >> 1) + 1;
  const numFrames = Math.max(0, Math.floor((padded.length - nFFT) / hopLength) + 1);

  const real = new Float32Array(numBins * numFrames);
  const imag = new Float32Array(numBins * numFrames);
  const fr = new Float32Array(nFFT);
  const fi = new Float32Array(nFFT);
  for (let t = 0; t < numFrames; t++) {
    const off = t * hopLength;
    for (let n = 0; n < nFFT; n++) {
      fr[n] = padded[off + n] * window[n];
      fi[n] = 0;
    }
    fft(fr, fi);
    for (let f = 0; f < numBins; f++) {
      real[f * numFrames + t] = fr[f];
      imag[f * numFrames + t] = fi[f];
    }
  }
  return { real, imag, numBins, numFrames };
}

export interface ISTFTOptions extends STFTOptions {
  length?: number; // trim/pad the reconstructed signal to this length
}

// Inverse STFT via weighted overlap-add. Consumes the one-sided spectrum
// produced by stft(); reconstructs the negative frequencies from conjugate
// symmetry for a real-valued output signal.
export function istft(
  real: Float32Array,
  imag: Float32Array,
  numFrames: number,
  opts: ISTFTOptions,
): Float32Array {
  const { nFFT, hopLength } = opts;
  const winLength = opts.winLength ?? nFFT;
  const center = opts.center ?? true;
  const numBins = (nFFT >> 1) + 1;
  if (real.length !== numBins * numFrames) {
    throw new Error(`istft: real length ${real.length} != numBins*numFrames ${numBins * numFrames}`);
  }
  const window = padWindow(opts.window ?? hannWindow(winLength), nFFT);

  const outLen = (numFrames - 1) * hopLength + nFFT;
  const out = new Float32Array(outLen);
  const windowSumSq = new Float32Array(outLen);

  const fr = new Float32Array(nFFT);
  const fi = new Float32Array(nFFT);
  for (let t = 0; t < numFrames; t++) {
    for (let f = 0; f < numBins; f++) {
      fr[f] = real[f * numFrames + t];
      fi[f] = imag[f * numFrames + t];
    }
    // Reconstruct the negative half via conjugate symmetry.
    for (let f = 1; f < numBins - 1; f++) {
      fr[nFFT - f] = fr[f];
      fi[nFFT - f] = -fi[f];
    }
    ifft(fr, fi);
    const off = t * hopLength;
    for (let n = 0; n < nFFT; n++) {
      out[off + n] += fr[n] * window[n];
      windowSumSq[off + n] += window[n] * window[n];
    }
  }
  // Normalize by overlapping window energy (WOLA). Guarded to avoid /0 at edges.
  for (let i = 0; i < outLen; i++) {
    const w = windowSumSq[i];
    if (w > 1e-10) out[i] /= w;
  }

  let final = out;
  if (center) final = out.subarray(nFFT >> 1, outLen - (nFFT >> 1));
  if (opts.length !== undefined) {
    const trimmed = new Float32Array(opts.length);
    trimmed.set(final.subarray(0, Math.min(final.length, opts.length)));
    final = trimmed;
  } else {
    final = new Float32Array(final);
  }
  return final;
}

function padReflect(signal: Float32Array, pad: number): Float32Array {
  const out = new Float32Array(signal.length + 2 * pad);
  for (let i = 0; i < pad; i++) out[pad - 1 - i] = signal[i + 1] ?? 0;
  out.set(signal, pad);
  const N = signal.length;
  for (let i = 0; i < pad; i++) out[pad + N + i] = signal[N - 2 - i] ?? 0;
  return out;
}

function padWindow(w: Float32Array, nFFT: number): Float32Array {
  if (w.length === nFFT) return w;
  const out = new Float32Array(nFFT);
  const pad = (nFFT - w.length) >> 1;
  out.set(w, pad);
  return out;
}
