import { Tensor } from "../frontend/tensor";

// Stateless activations. Layers-with-state (Linear, LayerNorm, ...) live in ./layers.

export function relu(x: Tensor): Tensor {
  return x.relu();
}

export function tanh(x: Tensor): Tensor {
  return x.tanh();
}

export function gelu(x: Tensor): Tensor {
  return x.gelu();
}

export function sigmoid(x: Tensor): Tensor {
  return x.sigmoid();
}

export function leaky_relu(x: Tensor, negativeSlope = 0.01): Tensor {
  return x.leaky_relu(negativeSlope);
}

export function silu(x: Tensor): Tensor {
  return x.silu();
}

export function softmax(x: Tensor, axis = -1): Tensor {
  return x.softmax(axis);
}
