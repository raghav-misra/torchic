import { init, shutdown } from "../../../src/index";
import { Kokoro, countParameters } from "./index";
import { defineTest } from "../../framework/define";
import type { RunContext, TestResult } from "../../framework/types";

type Backend = "workers" | "wasm" | "webgpu";

async function runShapes(backend: Backend, { log }: RunContext): Promise<TestResult> {
  const threadCount = backend === "webgpu" ? 1 : 4;
  await init({ backend, threadCount, memorySizeMB: 512 });
  try {
    log("constructing Kokoro model tree...");
    const m = new Kokoro();
    const total = countParameters(m);
    log(`total parameters: ${total.toLocaleString()} (~${(total / 1e6).toFixed(1)}M)`);

    const targetLo = 60_000_000;
    const targetHi = 160_000_000;
    log(`sanity target range: [${targetLo.toLocaleString()}, ${targetHi.toLocaleString()}]`);

    // Print param counts per top-level submodule.
    const buckets: Record<string, number> = {};
    const sd = m.state_dict();
    for (const key of Object.keys(sd)) {
      const top = key.split(".")[0];
      buckets[top] = (buckets[top] ?? 0) + sd[key].shape.reduce((a, b) => a * b, 1);
    }
    for (const [k, v] of Object.entries(buckets)) {
      log(`  ${k.padEnd(16)} ${v.toLocaleString()} (${((v / total) * 100).toFixed(1)}%)`);
    }

    const inRange = total >= targetLo && total <= targetHi;

    // Show a few key parameter shapes so we can eyeball them against a real
    // checkpoint. These are the ones we need to match exactly.
    const spot: string[] = [
      "bert.embeddings.word_embeddings.weight",
      "bert_encoder.weight",
      "text_encoder.embedding.weight",
      "text_encoder.lstm.fwd.weight_ih",
      "predictor.lstm.fwd.weight_ih",
      "predictor.duration_proj.weight",
      "predictor.F0_proj.weight",
      "decoder.encode.conv1.weight",
      "decoder.decode.3.pool.weight",
      "decoder.asr_res.0.weight",
      "decoder.generator.m_source.l_linear.weight",
      "decoder.generator.ups.0.weight",
      "decoder.generator.resblocks.0.convs1.0.weight",
      "decoder.generator.resblocks.0.alpha1.0",
      "decoder.generator.conv_post.weight",
    ];
    log("spot-check shapes:");
    for (const k of spot) {
      const t = sd[k];
      log(`  ${k.padEnd(52)} ${t ? t.shape.join("x") : "(MISSING)"}`);
    }

    return {
      pass: inRange,
      message: inRange
        ? `${total.toLocaleString()} params (${Object.keys(sd).length} tensors)`
        : `${total.toLocaleString()} params outside [${targetLo}, ${targetHi}]`,
    };
  } finally {
    shutdown();
  }
}

defineTest<Backend>({
  name: "Kokoro: model skeleton + param count",
  paramName: "backend",
  params: ["workers"],
  description:
    "Instantiates the Kokoro model tree and reports param counts. No weights loaded yet; this exists so the demo review can quickly spot missing/mis-shaped modules against the real 82M checkpoint.",
  runner: runShapes,
});
