#!/usr/bin/env python3
"""
export_tinyllama_q8_0.py: Cuantiza y exporta TinyLlama-1.1B a formato modular Q8_0.
Genera un archivo binario por capa (layer_00.bin .. layer_21.bin), embed_tokens,
final_norm y lm_head para inferencia de ultra-alto rendimiento en silicio Kepler.
"""

import os
import sys
import json
import struct
import ml_dtypes
import time
import numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download

REPO_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "models/tinyllama_q8_0"

def quantize_matrix_q8_0(W_fp32):
    """
    Cuantiza una matriz 2D (M, K) a formato block_q8_0 (36 bytes por bloque de 32 elementos).
    Formato: float32 delta (4B) + 32 x int8 qs (32B).
    """
    M, K = W_fp32.shape
    assert K % 32 == 0, f"K={K} must be a multiple of 32"
    blocks = W_fp32.reshape(-1, 32)
    max_vals = np.max(np.abs(blocks), axis=1)
    deltas = (max_vals / 127.0).astype(np.float32)
    inv_deltas = np.where(deltas > 0, 1.0 / deltas, 0.0)
    qs = np.clip(np.round(blocks * inv_deltas[:, None]), -128, 127).astype(np.int8)

    dt = np.dtype([('d', '<f4'), ('qs', 'i1', (32,))])
    packed = np.empty(len(deltas), dtype=dt)
    packed['d'] = deltas
    packed['qs'] = qs
    return packed.tobytes()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Target directory: {OUTPUT_DIR}")

    # 1. Localizar o descargar model.safetensors
    safetensors_path = None
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        safetensors_path = sys.argv[1]
    else:
        print(f"[*] Locating/downloading {REPO_ID} model.safetensors via huggingface_hub...")
        safetensors_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")

    print(f"[+] Using weights from: {safetensors_path}")

    t_start = time.time()
    with safe_open(safetensors_path, framework="numpy", device="cpu") as f:
        tensor_keys = list(f.keys())
        print(f"[+] Loaded safetensors index with {len(tensor_keys)} tensors.")

        # 2. Exportar Embeddings (FP32 en Host para lookup O(1) de 8 KB por token)
        embed_key = "model.embed_tokens.weight"
        print(f"[*] Exporting Host Embeddings from '{embed_key}'...")
        w_embed = f.get_tensor(embed_key).astype(np.float32)
        print(f"    Shape: {w_embed.shape}, Dtype: {w_embed.dtype}")
        embed_bin_path = os.path.join(OUTPUT_DIR, "embed_tokens.bin")
        w_embed.tofile(embed_bin_path)
        print(f"    Saved {embed_bin_path} ({os.path.getsize(embed_bin_path) / (1024*1024):.2f} MB)")

        # 3. Exportar RMSNorm Final (FP32, 2048 floats = 8192 bytes)
        norm_key = "model.norm.weight"
        print(f"[*] Exporting Final RMSNorm from '{norm_key}'...")
        w_norm = f.get_tensor(norm_key).astype(np.float32)
        final_norm_path = os.path.join(OUTPUT_DIR, "final_norm.bin")
        w_norm.tofile(final_norm_path)
        print(f"    Saved {final_norm_path} ({os.path.getsize(final_norm_path)} bytes)")

        # 4. Exportar LM-Head (Q8_0, 32000 x 2048 = ~70.3 MB)
        lm_head_key = "lm_head.weight"
        print(f"[*] Quantizing and Exporting LM-Head from '{lm_head_key}'...")
        w_lm_head = f.get_tensor(lm_head_key).astype(np.float32)
        lm_head_bytes = quantize_matrix_q8_0(w_lm_head)
        lm_head_path = os.path.join(OUTPUT_DIR, "lm_head.bin")
        with open(lm_head_path, "wb") as out_f:
            out_f.write(lm_head_bytes)
        print(f"    Saved {lm_head_path} ({os.path.getsize(lm_head_path) / (1024*1024):.2f} MB)")

        # 5. Exportar las 22 capas Transformer
        print("[*] Processing 22 Transformer Decoder Layers...")
        num_layers = 22
        for l in range(num_layers):
            layer_prefix = f"model.layers.{l}"
            layer_filename = os.path.join(OUTPUT_DIR, f"layer_{l:02d}.bin")
            t_l0 = time.time()

            # RMSNorms en FP32
            w_attn_norm = f.get_tensor(f"{layer_prefix}.input_layernorm.weight").astype(np.float32)
            w_ffn_norm  = f.get_tensor(f"{layer_prefix}.post_attention_layernorm.weight").astype(np.float32)

            # Matrices lineales a cuantizar en Q8_0
            w_q    = f.get_tensor(f"{layer_prefix}.self_attn.q_proj.weight").astype(np.float32)
            w_k    = f.get_tensor(f"{layer_prefix}.self_attn.k_proj.weight").astype(np.float32)
            w_v    = f.get_tensor(f"{layer_prefix}.self_attn.v_proj.weight").astype(np.float32)
            w_o    = f.get_tensor(f"{layer_prefix}.self_attn.o_proj.weight").astype(np.float32)
            w_gate = f.get_tensor(f"{layer_prefix}.mlp.gate_proj.weight").astype(np.float32)
            w_up   = f.get_tensor(f"{layer_prefix}.mlp.up_proj.weight").astype(np.float32)
            w_down = f.get_tensor(f"{layer_prefix}.mlp.down_proj.weight").astype(np.float32)

            # Cuantizar a Q8_0
            q_bytes_Wq    = quantize_matrix_q8_0(w_q)
            q_bytes_Wk    = quantize_matrix_q8_0(w_k)
            q_bytes_Wv    = quantize_matrix_q8_0(w_v)
            q_bytes_Wo    = quantize_matrix_q8_0(w_o)
            q_bytes_Wgate = quantize_matrix_q8_0(w_gate)
            q_bytes_Wup   = quantize_matrix_q8_0(w_up)
            q_bytes_Wdown = quantize_matrix_q8_0(w_down)

            # Escribir archivo de capa
            with open(layer_filename, "wb") as out_l:
                # 1. RMSNorm 1 (8192 B)
                out_l.write(w_attn_norm.tobytes())
                # 2. RMSNorm 2 (8192 B)
                out_l.write(w_ffn_norm.tobytes())
                # 3. Pesos Q8_0 contiguos (47.25 MB)
                out_l.write(q_bytes_Wq)
                out_l.write(q_bytes_Wk)
                out_l.write(q_bytes_Wv)
                out_l.write(q_bytes_Wo)
                out_l.write(q_bytes_Wgate)
                out_l.write(q_bytes_Wup)
                out_l.write(q_bytes_Wdown)

            t_l1 = time.time()
            sz_mb = os.path.getsize(layer_filename) / (1024 * 1024)
            print(f"    [Layer {l:02d}/21] Saved {layer_filename} ({sz_mb:.2f} MB in {t_l1 - t_l0:.2f} s)")

        # 6. Metadatos de configuración
        config_data = {
            "model_name": "TinyLlama-1.1B-Chat-v1.0",
            "vocab_size": 32000,
            "dim": 2048,
            "hidden_dim": 5632,
            "num_layers": 22,
            "num_heads": 32,
            "num_kv_heads": 4,
            "head_dim": 64,
            "max_seq_len": 2048,
            "norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "rope_mode": 0,
            "bos_token_id": 1,
            "eos_token_id": 2
        }
        cfg_out_path = os.path.join(OUTPUT_DIR, "model_config.json")
        with open(cfg_out_path, "w", encoding="utf-8") as cfg_f:
            json.dump(config_data, cfg_f, indent=2)
        print(f"[+] Saved metadata to {cfg_out_path}")

    t_total = time.time() - t_start
    print(f"\n====================================================================")
    print(f"[+] ALL 22 LAYERS AND WEIGHTS EXPORTED SUCCESSFULLY IN {t_total:.2f} SECONDS!")
    print(f"====================================================================")

if __name__ == "__main__":
    main()
