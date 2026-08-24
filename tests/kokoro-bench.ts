import { Tensor, init, noGrad, nn } from "../src/index";
import { Kokoro } from "./demos/kokoro/index";
import SAMPLES_JSON from "./demos/kokoro/samples.json";

interface Sample { label: string; phonemes: string; ids: number[]; voice_idx: number }
const SAMPLES: Record<string, Sample> = SAMPLES_JSON;

const SAMPLE_RATE = 24000;

const status = document.getElementById("status")!;
const logEl = document.getElementById("log")!;
function log(msg: string): void {
  logEl.textContent += msg + "\n";
  console.log(`[bench] ${msg}`);
}

// Waits for the referenced <input type="file"> to receive a File. Puppeteer
// uploads to these persistent inputs via `elementHandle.uploadFile(path)`,
// which is the reliable path in headless — the filechooser event doesn't
// fire for JS-synthesized `.click()` in headless Chromium.
function waitForFile(inputId: string): Promise<File> {
  return new Promise((resolve, reject) => {
    const input = document.getElementById(inputId) as HTMLInputElement | null;
    if (!input) return reject(new Error(`missing input #${inputId}`));
    if (input.files && input.files.length > 0) {
      resolve(input.files[0]);
      return;
    }
    input.addEventListener("change", () => {
      const f = input.files?.[0];
      if (f) resolve(f);
      else reject(new Error(`#${inputId}: no file`));
    }, { once: true });
  });
}

function tensorPeak(audio: Float32Array): { peak: number; nans: number } {
  let peak = 0;
  let nans = 0;
  for (const v of audio) {
    if (Number.isNaN(v)) nans++;
    const a = Math.abs(v);
    if (a > peak) peak = a;
  }
  return { peak, nans };
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const sampleKey = params.get("sample") ?? "pangram";
  const backend = (params.get("backend") ?? "webgpu") as "webgpu" | "workers";
  const memorySizeMB = parseInt(params.get("memory") ?? "1536", 10);

  const sample = SAMPLES[sampleKey];
  if (!sample) throw new Error(`unknown sample '${sampleKey}'. available: ${Object.keys(SAMPLES).join(", ")}`);

  log(`config: sample=${sampleKey} backend=${backend} memory=${memorySizeMB}MB`);
  log(`sample: '${sample.label}' (${sample.ids.length} tokens)`);

  status.textContent = "init";
  await init({ backend, memorySizeMB });
  log(`backend ready`);

  const model = new Kokoro();
  model.eval();
  log(`Kokoro built (~82M params)`);

  status.textContent = "await-model";
  log(`waiting for model .safetensors file (upload to #model-input)...`);
  const modelFile = await waitForFile("model-input");
  log(`reading ${modelFile.name} (${(modelFile.size / 1e6).toFixed(1)} MB)`);
  const modelBuf = await modelFile.arrayBuffer();
  const parsed = nn.parseSafetensors(modelBuf);
  const { missing, unexpected } = model.load_safetensors(parsed, { strict: false });
  log(`load: ${missing.length} missing, ${unexpected.length} unexpected keys`);
  if (missing.length > 0 || unexpected.length > 0) throw new Error(`state_dict mismatch`);

  status.textContent = "await-voice";
  log(`waiting for voice .safetensors file (upload to #voice-input)...`);
  const voiceFile = await waitForFile("voice-input");
  log(`reading ${voiceFile.name} (${(voiceFile.size / 1024).toFixed(1)} KB)`);
  const voiceBuf = await voiceFile.arrayBuffer();
  const voice = nn.parseSafetensors(voiceBuf)["voice"];
  if (!voice) throw new Error(`voice safetensors missing 'voice' key`);

  const rows = voice.shape[0];
  const styleSize = voice.shape[voice.shape.length - 1];
  const idx = Math.min(Math.max(sample.voice_idx, 0), rows - 1);
  const perRow = voice.data.length / rows;
  const refData = voice.data.subarray(idx * perRow, (idx + 1) * perRow);
  const refS = Tensor.fromData(Array.from(refData), [1, styleSize]);
  const inputIds = Tensor.fromData(sample.ids.slice(), [1, sample.ids.length]);

  status.textContent = "synth";
  log(`synthesizing...`);
  const started = performance.now();
  const { audio, predDur } = await noGrad(() => model.forward(inputIds, refS, { speed: 1 }));
  const elapsed = (performance.now() - started) / 1000;

  const audioSec = audio.length / SAMPLE_RATE;
  const rtf = elapsed / audioSec;
  const { peak, nans } = tensorPeak(audio);
  const durSum = predDur.reduce((a, b) => a + b, 0);

  log(`durations: ${predDur.join(", ")}`);
  log(`audio: ${audio.length} samples (${audioSec.toFixed(2)}s @ ${SAMPLE_RATE}Hz)`);
  log(`synthesis took ${elapsed.toFixed(2)}s → RTF ${rtf.toFixed(3)}`);
  log(`peak amplitude: ${peak.toExponential(3)}${nans ? ` (${nans} NaN)` : ""}`);

  const result = {
    sample: sampleKey,
    backend,
    tokens: sample.ids.length,
    audioSamples: audio.length,
    audioSec,
    elapsedSec: elapsed,
    rtf,
    peak,
    nans,
    durSum,
    predDur,
  };
  console.log(`__RESULT__${JSON.stringify(result)}`);
  status.textContent = "done";
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  log(`error: ${msg}`);
  console.log(`__ERROR__${msg}`);
  status.textContent = "error";
});
