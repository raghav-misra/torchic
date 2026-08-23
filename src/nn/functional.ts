import { Tensor } from "../frontend/tensor";

// Stateless activations. Layers-with-state (Linear, LayerNorm, ...) live in ./layers.

export function relu(x: Tensor): Tensor {
  return x.relu();
}

export function tanh(x: Tensor): Tensor {
  return x.tanh();
}

export function softmax(x: Tensor, axis = -1): Tensor {
  return x.softmax(axis);
}
