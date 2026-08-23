import { Tensor, init, shutdown, noGrad, nn } from "../../../src/index";
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
  voice: nn.SafetensorsEntry | null;
  weightsLoaded: boolean;
  audio: Float32Array | null;
  audioCtx: AudioContext | null;
}

const state: State = {
  model: null,
  refS: null,
  voice: null,
  weightsLoaded: false,
  audio: null,
  audioCtx: null,
};

function pickFile(accept: string): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.onchange = () => resolve(input.files?.[0] ?? null);
    (input as HTMLInputElement & { oncancel?: () => void }).oncancel = () => resolve(null);
    input.click();
  });
}

async function initBackend(ctx: FreeformContext): Promise<void> {

  const memorySizeMB = 1536;
  ctx.log(`initializing webgpu backend (heap ${memorySizeMB} MB)...`);
  await init({ backend: "webgpu", memorySizeMB });
  ctx.log("webgpu ready.");
  ctx.log("building Kokoro module tree (~82M random-init params)...");
  state.model = new Kokoro();
  state.model.eval();
  state.refS = Tensor.randn([1, 256]);
  ctx.log("model ready.");
  ctx.enable("load_model");
  ctx.enable("load_voice");
  ctx.enable("synthesize_hello");
  ctx.disable("init");
}

async function loadModel(ctx: FreeformContext): Promise<void> {
  if (!state.model) return;
  const file = await pickFile(".safetensors");
  if (!file) {
    ctx.log("no file selected.");
    return;
  }
  ctx.log(`reading ${file.name} (${(file.size / 1e6).toFixed(1)} MB)...`);
  const buf = await file.arrayBuffer();
  const parsed = nn.parseSafetensors(buf);
  ctx.log(`parsed ${Object.keys(parsed).length} tensors from checkpoint.`);
  const { missing, unexpected } = state.model.load_safetensors(parsed, { strict: false });
  ctx.log(`load: ${missing.length} missing, ${unexpected.length} unexpected keys.`);
  if (missing.length > 0) ctx.log(`  first missing: ${missing.slice(0, 5).join(", ")}`);
  if (unexpected.length > 0) ctx.log(`  first unexpected: ${unexpected.slice(0, 5).join(", ")}`);
  state.weightsLoaded = true;
  ctx.log("model weights loaded.");
}

async function loadVoice(ctx: FreeformContext): Promise<void> {
  const file = await pickFile(".safetensors");
  if (!file) {
    ctx.log("no file selected.");
    return;
  }
  ctx.log(`reading voice ${file.name} (${(file.size / 1024).toFixed(1)} KB)...`);
  const buf = await file.arrayBuffer();
  const parsed = nn.parseSafetensors(buf);
  const voice = parsed["voice"];
  if (!voice) {
    ctx.log(`expected 'voice' key; got: ${Object.keys(parsed).join(", ")}`);
    return;
  }
  ctx.log(`voice tensor shape [${voice.shape.join(", ")}], ${voice.data.length} floats.`);
  state.voice = voice;
}

// Kokoro voice packs are [511, 1, 256] — one style vector per phoneme count.
// Reference kokoro-js uses voice[len(input_ids) - 1].
function pickRefFromVoice(voice: nn.SafetensorsEntry, tokenCount: number): Tensor {
  const rows = voice.shape[0];
  const styleSize = voice.shape[voice.shape.length - 1];
  const idx = Math.min(Math.max(tokenCount - 1, 0), rows - 1);
  const perRow = voice.data.length / rows;
  const slice = voice.data.subarray(idx * perRow, (idx + 1) * perRow);
  return Tensor.fromData(Array.from(slice), [1, styleSize]);
}

async function synthesize(ctx: FreeformContext, sample: keyof typeof SAMPLES): Promise<void> {
  if (!state.model || !state.refS) {
    ctx.log("model not initialized — click Init first.");
    return;
  }
  const info = SAMPLES[sample];
  ctx.log(`synthesizing '${info.label}' (${info.ids.length} tokens)...`);
  const inputIds = Tensor.fromData(info.ids.slice(), [1, info.ids.length]);

  const usingRealWeights = state.weightsLoaded;
  const ref = state.voice && usingRealWeights
    ? pickRefFromVoice(state.voice, info.ids.length)
    : state.refS;
  const speed = usingRealWeights ? 1 : 100;
  ctx.log(
    `mode: ${usingRealWeights ? "real weights" : "random weights (noise expected)"}, ref_s=${
      state.voice ? "voice pack" : "randn"
    }, speed=${speed}`,
  );

  const started = performance.now();
  // No trackTensors here on purpose: it adds every allocation to a Set which
  // holds strong refs and prevents FinalizationRegistry from firing on
  // intermediates, so the working set balloons to *all* tensors ever
  // allocated in the pass. eval() already suppresses the autograd tape, so
  // dropping refs + letting GC recycle is the right strategy.
  const { audio, predDur } = await noGrad(() =>
    state.model!.forward(inputIds, ref, { speed }),
  );
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
  state.voice = null;
  state.weightsLoaded = false;
  if (state.audioCtx) {
    state.audioCtx.close();
    state.audioCtx = null;
  }
  shutdown();
  ctx.disable("load_model");
  ctx.disable("load_voice");
  ctx.disable("synthesize_hello");
  ctx.disable("play");
  ctx.enable("init");
}

defineFreeform({
  name: "Kokoro: end-to-end synthesis",
  description:
    "State machine: init backend + build Kokoro, optionally load a converted safetensors checkpoint + voice pack, synthesize a fixed phoneme sequence, play the resulting PCM. Without real weights the pipeline still runs but produces noise (speed=100 caps durations to 1 per phoneme so intermediates fit in memory).",
  actions: [
    { id: "init", label: "1. Init backend + build model", run: initBackend },
    { id: "load_model", label: "2a. Load model.safetensors", disabled: true, run: loadModel },
    { id: "load_voice", label: "2b. Load voice.safetensors", disabled: true, run: loadVoice },
    { id: "synthesize_hello", label: `3. Synthesize "${SAMPLES.hello_world.label}"`, disabled: true, run: (c) => synthesize(c, "hello_world") },
    { id: "play", label: "4. Play last synthesis", disabled: true, run: async (c) => play(c) },
    { id: "cleanup", label: "Shutdown", run: async (c) => cleanup(c) },
  ],
});
