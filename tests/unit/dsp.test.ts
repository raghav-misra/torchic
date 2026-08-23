import { describe, it, expect } from "vitest";
import { fft, ifft, stft, istft, hannWindow } from "../../src/dsp";

describe("FFT", () => {
  it("round-trips a random signal", () => {
    const n = 64;
    const rand = () => (Math.random() - 0.5) * 2;
    const real = new Float32Array(Array.from({ length: n }, rand));
    const imag = new Float32Array(Array.from({ length: n }, rand));
    const r0 = new Float32Array(real);
    const i0 = new Float32Array(imag);
    fft(real, imag);
    ifft(real, imag);
    for (let i = 0; i < n; i++) {
      expect(Math.abs(real[i] - r0[i])).toBeLessThan(1e-4);
      expect(Math.abs(imag[i] - i0[i])).toBeLessThan(1e-4);
    }
  });

  it("returns correct spectrum for a single sinusoid", () => {
    const n = 16;
    const k = 3; // frequency bin
    const real = new Float32Array(n);
    const imag = new Float32Array(n);
    for (let i = 0; i < n; i++) real[i] = Math.cos((2 * Math.PI * k * i) / n);
    fft(real, imag);
    for (let i = 0; i < n; i++) {
      const mag = Math.hypot(real[i], imag[i]);
      if (i === k || i === n - k) {
        expect(mag).toBeCloseTo(n / 2, 3);
      } else {
        expect(mag).toBeLessThan(1e-3);
      }
    }
  });
});

describe("STFT ↔ ISTFT round-trip", () => {
  it("reconstructs a short sine with Hann window (center=true)", () => {
    const nFFT = 128;
    const hop = 32;
    const len = 512;
    const x = new Float32Array(len);
    for (let i = 0; i < len; i++) x[i] = Math.sin((2 * Math.PI * 5 * i) / len);
    const { real, imag, numFrames } = stft(x, { nFFT, hopLength: hop, center: true });
    const y = istft(real, imag, numFrames, { nFFT, hopLength: hop, center: true, length: len });
    // Overlap-add reconstruction tolerance for Hann + 75% overlap ~ 1e-5.
    for (let i = 0; i < len; i++) {
      expect(Math.abs(y[i] - x[i])).toBeLessThan(1e-3);
    }
  });

  it("hann window is periodic (sums to 1 with 50% overlap)", () => {
    const w = hannWindow(16, true);
    // Periodic hann: half + shifted half sums to 1 everywhere.
    for (let i = 0; i < 8; i++) {
      expect(w[i] + w[i + 8]).toBeCloseTo(1, 5);
    }
  });
});
