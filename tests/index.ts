import "./suites/wasm-parity";
import "./suites/webgpu-parity";
import "./suites/matmul-bench";
import "./suites/layernorm-gelu";
import "./suites/bmm";
import "./suites/attention";
import "./suites/conv1d";
import "./suites/lstm";
import "./suites/shape-ops";
import "./suites/safetensors";
import "./demos/kokoro/skeleton.test";
import "./demos/makemore";
import "./console";
import { mount } from "./framework/render";

const root = document.getElementById("root");
if (!root) throw new Error("no #root element");
mount(root);
