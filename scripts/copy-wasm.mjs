import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(
  here,
  "..",
  "crates",
  "torchic-kernels",
  "target",
  "wasm32-unknown-unknown",
  "release",
  "torchic_kernels.wasm",
);
const dst = resolve(here, "..", "src", "backend", "wasm", "kernels.wasm");

await mkdir(dirname(dst), { recursive: true });
await copyFile(src, dst);
console.log(`copied ${src} -> ${dst}`);
