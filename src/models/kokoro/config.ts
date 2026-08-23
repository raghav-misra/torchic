// Kokoro-82M config. Mirrors hexgrad/Kokoro-82M/config.json exactly.
// Keeping the field names in snake_case matches the checkpoint side, which
// simplifies state_dict key alignment.

export interface KokoroISTFTNetConfig {
  upsample_kernel_sizes: number[];
  upsample_rates: number[];
  gen_istft_hop_size: number;
  gen_istft_n_fft: number;
  resblock_dilation_sizes: number[][];
  resblock_kernel_sizes: number[];
  upsample_initial_channel: number;
}

export interface KokoroPLBERTConfig {
  hidden_size: number;
  num_attention_heads: number;
  intermediate_size: number;
  max_position_embeddings: number;
  num_hidden_layers: number;
  dropout: number;
}

export interface KokoroConfig {
  istftnet: KokoroISTFTNetConfig;
  dim_in: number;
  dropout: number;
  hidden_dim: number;
  max_conv_dim: number;
  max_dur: number;
  multispeaker: boolean;
  n_layer: number;
  n_mels: number;
  n_token: number;
  style_dim: number;
  text_encoder_kernel_size: number;
  plbert: KokoroPLBERTConfig;
  vocab: Record<string, number>;
}

// The full Kokoro-82M v1.0 config. Matches
// https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json
export const KOKORO_CONFIG: KokoroConfig = {
  istftnet: {
    upsample_kernel_sizes: [20, 12],
    upsample_rates: [10, 6],
    gen_istft_hop_size: 5,
    gen_istft_n_fft: 20,
    resblock_dilation_sizes: [
      [1, 3, 5],
      [1, 3, 5],
      [1, 3, 5],
    ],
    resblock_kernel_sizes: [3, 7, 11],
    upsample_initial_channel: 512,
  },
  dim_in: 64,
  dropout: 0.2,
  hidden_dim: 512,
  max_conv_dim: 512,
  max_dur: 50,
  multispeaker: true,
  n_layer: 3,
  n_mels: 80,
  n_token: 178,
  style_dim: 128,
  text_encoder_kernel_size: 5,
  plbert: {
    hidden_size: 768,
    num_attention_heads: 12,
    intermediate_size: 2048,
    max_position_embeddings: 512,
    num_hidden_layers: 12,
    dropout: 0.1,
  },
  vocab: {}, // populated at load time from vocab.ts if needed for G2P output mapping
};
