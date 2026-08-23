import "./suites/wasm-parity";
import "./suites/webgpu-parity";
import "./suites/matmul-bench";
import "./demos/makemore";
import "./console";
import { mount } from "./framework/render";

const root = document.getElementById("root");
if (!root) throw new Error("no #root element");
mount(root);
