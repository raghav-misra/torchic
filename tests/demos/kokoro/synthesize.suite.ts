import { Tensor, init, shutdown, noGradSync } from "../../../src/index";
import { defineFreeform } from "../../framework/define";
import type { FreeformContext } from "../../framework/types";
import { Kokoro } from "./index";

const SAMPLE_RATE = 24000;

// Predefined phoneme sequences with wrapping zero tokens.
// IDs are pulled from KOKORO_CONFIG.vocab (see kokoro/config.json).
const SAMPLES = {
  hello_world: {
    label: "hˈɛloʊ wˈɜɹld",
    ids: [0, 50, 156, 86, 54, 57, 135, 16, 65, 156, 87, 123, 54, 46, 0],
  },
} as const;

interface State {
  model: Kokoro | null;
  refS: Tensor | null;
  audio: Float32Array | null;
  audioCtx: AudioContext | null;
}

const state: State = { model: null, refS: null, audio: null, audioCtx: null };

async function initBackend(ctx: FreeformContext): Promise<void> {
  ctx.log("initializing webgpu backend...");
  try {
    await init({ backend: "webgpu", memorySizeMB: 512 });
    ctx.log("webgpu ready.");
  } catch (e) {
    ctx.log(`webgpu init failed (${String(e)}), falling back to wasm.`);
    await init({ backend: "wasm", threadCount: 4, memorySizeMB: 512 });
    ctx.log("wasm ready.");
  }
  ctx.log("building Kokoro module tree (~82M random-init params)...");
  state.model = new Kokoro();
  state.model.eval();
  state.refS = Tensor.randn([1, 256]);
  ctx.log("model ready.");
  ctx.enable("synthesize_hello");
  ctx.disable("init");
}

async function synthesize(ctx: FreeformContext, sample: keyof typeof SAMPLES): Promise<void> {
  if (!state.model || !state.refS) {
    ctx.log("model not initialized — click Init first.");
    return;
  }
  const info = SAMPLES[sample];
  ctx.log(`synthesizing '${info.label}' (${info.ids.length} tokens)...`);
  const inputIds = Tensor.fromData(info.ids.slice(), [1, info.ids.length]);

  const started = performance.now();
  const { audio, predDur } = await noGradSync(() => state.model!.forward(inputIds, state.refS!));
  const elapsed = (performance.now() - started) / 1000;
  const audioSec = audio.length / SAMPLE_RATE;
  const rtf = elapsed / audioSec;

  ctx.log(`durations: ${predDur.join(", ")}`);
  ctx.log(`audio: ${audio.length} samples (${audioSec.toFixed(2)}s @ ${SAMPLE_RATE}Hz)`);
  ctx.log(`synthesis took ${elapsed.toFixed(2)}s → RTF ${rtf.toFixed(3)}`);

  let peak = 0;
  let nans = 0;
  for (const v of audio) {
    if (Number.isNaN(v)) nans++;
    const a = Math.abs(v);
    if (a > peak) peak = a;
  }
  ctx.log(`peak amplitude: ${peak.toExponential(2)}${nans ? ` (${nans} NaN samples)` : ""}`);
  if (nans > 0) ctx.log("note: NaNs likely from random-init weights; load a real checkpoint to hear anything.");

  state.audio = audio;
  ctx.enable("play");
}

function play(ctx: FreeformContext): void {
  if (!state.audio) {
    ctx.log("no audio to play — synthesize first.");
    return;
  }
  const AC = window.AudioContext ?? (window as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AC) {
    ctx.log("browser has no AudioContext.");
    return;
  }
  state.audioCtx ??= new AC({ sampleRate: SAMPLE_RATE });
  // Clamp NaN / out-of-range samples so playback doesn't error out with random weights.
  const clean = new Float32Array(state.audio.length);
  for (let i = 0; i < state.audio.length; i++) {
    const v = state.audio[i];
    clean[i] = Number.isNaN(v) ? 0 : Math.max(-1, Math.min(1, v));
  }
  const buffer = state.audioCtx.createBuffer(1, clean.length, SAMPLE_RATE);
  buffer.copyToChannel(clean, 0);
  const src = state.audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(state.audioCtx.destination);
  src.start();
  ctx.log(`playing ${(clean.length / SAMPLE_RATE).toFixed(2)}s of audio...`);
}

function cleanup(ctx: FreeformContext): void {
  ctx.log("shutting down backend + clearing state.");
  state.audio = null;
  state.model = null;
  state.refS = null;
  if (state.audioCtx) {
    state.audioCtx.close();
    state.audioCtx = null;
  }
  shutdown();
  ctx.disable("synthesize_hello");
  ctx.disable("play");
  ctx.enable("init");
}

defineFreeform({
  name: "Kokoro: end-to-end synthesis",
  description:
    "State machine: init a backend + instantiate Kokoro with random weights, synthesize a fixed phoneme sequence, play the resulting PCM. Real audio requires loading the converted safetensors checkpoint (see scripts/convert-kokoro-checkpoint.py); random weights sound like noise but exercise the full forward pipeline.",
  actions: [
    { id: "init", label: "1. Init backend + build model", run: initBackend },
    { id: "synthesize_hello", label: `2. Synthesize "${SAMPLES.hello_world.label}"`, disabled: true, run: (c) => synthesize(c, "hello_world") },
    { id: "play", label: "3. Play last synthesis", disabled: true, run: async (c) => play(c) },
    { id: "cleanup", label: "Shutdown", run: async (c) => cleanup(c) },
  ],
});
