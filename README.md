# 🏴‍☠️ Kepler CUDA Native LLM Runner
### Direct Hardware Execution of TinyLlama-1.1B on NVIDIA Kepler GK107 (sm_30) via Linux Nouveau DRM

[![Silicon Target: Kepler sm_30](https://img.shields.io/badge/Silicon-Kepler%20GK107%20sm__30-76b900.svg)](https://en.wikipedia.org/wiki/Kepler_(microarchitecture))
[![VRAM: 2 GB GDDR5](https://img.shields.io/badge/VRAM-2048%20MB%20GDDR5-blue.svg)]()
[![Kernel Driver: Linux Nouveau DRM](https://img.shields.io/badge/Driver-Linux%20Nouveau%20DRM%20(Open%20Source)-orange.svg)](https://nouveau.freedesktop.org/)
[![Hardware Validation: 20/20 Suites PASS](https://img.shields.io/badge/Hardware%20Validation-20%2F20%20Suites%20PASS%20(100%25)-brightgreen.svg)]()
[![Turbo Clock: P0 925 MHz](https://img.shields.io/badge/Clock-P0%20Turbo%20925%20MHz-red.svg)]()

> *"Good artists copy, great artists steal."* — Steve Jobs / *Pirates of Silicon Valley*

---

## 🧭 Executive Summary & The Impossible Silicon Challenge

In 2026, modern AI frameworks (PyTorch 2.x, vLLM, TensorRT-LLM, CUDA 12.x) mandate modern GPUs with Tensor Cores, FP8/FP16 native tensor engines, and unified virtual memory. Architectures prior to Pascal (sm_60) have been officially deprecated and dropped by NVIDIA.

The **NVIDIA GeForce GT 750M** (Kepler GK107 sm_30, 384 CUDA cores, 2048 MB GDDR5 memory) powering the **Apple MacBook Pro (Retina, 15-inch, Mid 2014)** was declared obsolete:
- **No official modern CUDA support.**
- **No Tensor Cores or INT8/FP16 native tensor math.**
- **Strict 2 GB VRAM memory envelope** (where ~300 MB is reserved by the 2880x1800 Retina display framebuffer).
- **Default open-source Nouveau driver locked at baseline idle frequencies** (405 MHz core, 1560 MHz GDDR5).

**This repository proves that complete, autoregressive Transformer LLM inference (TinyLlama-1.1B) can be achieved in 100% pure native C and bare-metal Kepler machine code, communicating directly with the Linux Nouveau DRM kernel without proprietary blobs or external runtimes.**

---

## 🏛️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        USER / TERMINAL REPL                            │
│                 ./bin/tinyllama_runner -n 250 "Prompt"                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Pure C99
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   SENTENCEPIECE BPE TOKENIZER IN C                     │
│    - Byte-level prefix fallback (0xE2 0x96 0x81 -> Space)              │
│    - ChatML / Zephyr Formatting Template                               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Token IDs [0 .. 32000]
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 TINYLLAMA AUTOREGRESSIVE RUNNER CORE                   │
│    - Token Embeddings on Host (250 MB FP32)                            │
│    - Static Resident Weights in VRAM: 1154.16 MB (~1.15 GB)            │
│    - Multilayer Unified KV-Cache in VRAM: 44.0 MB (pos=0..1024)        │
│    - Dynamic On-Demand DVFS: P0 Turbo (925 MHz) <-> P5 Idle (405 MHz)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ libcuda.so Native Shim
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    NOUVEAU DRM DRIVER SHIM (C99)                       │
│    - DRM_IOCTL_NOUVEAU_CHANNEL_ALLOC (PFIFO Pushbuffer)                │
│    - DRM_IOCTL_NOUVEAU_GEM_NEW & GEM_CPU_PREP / FINI (DMA BOs)         │
│    - Direct QMD sm_30 Construction & Constant Buffers (CB0, CB2)       │
│    - Synchronous Fence Wait via dma_fence / nvkm                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ PCIe Gen3 x8/x16
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             NVIDIA GEFORCE GT 750M SILICON (KEPLER GK107)              │
│    2 SMX Units | 384 CUDA Cores | 65,536 Registers | 2 GB GDDR5        │
│                                                                        │
│  [rms_norm.cubin]       -> Layer Norm (FP32)                           │
│  [gemv_q8_0.cubin]      -> Q, K, V Projections (W8A32 dot-product)     │
│  [rope_kcache.cubin]    -> Rotary Position Embeddings (RoPE)           │
│  [attention_decode.cubin] -> Softmax & KV Reduction over Context       │
│  [swiglu.cubin]         -> Feed-Forward Network Activation             │
│  [residual_add.cubin]   -> Elementwise Accumulation                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Chronology of the Silicon Odyssey

### Phase 1: Reverse-Engineering Nouveau DRM & The Zero-Blob Shim
Modern CUDA applications rely on `nvidia.ko` and `libcuda.so.1`. Under Linux with Nouveau, no proprietary user-space driver exists. 
We reverse-engineered the raw DRM IOCTLs of the `nouveau` kernel module:
1. Allocated a 3D channel via `DRM_IOCTL_NOUVEAU_CHANNEL_ALLOC`.
2. Created Buffer Objects (BOs) in VRAM and GTT via `DRM_IOCTL_NOUVEAU_GEM_NEW`.
3. Mapped the pushbuffer and subchannel engines (`NVE4_COMPUTE`).
4. Constructed Kepler Queue Meta-Data (QMD sm_30) descriptors containing kernel function entry pointers, register counts (`regs`), barrier counts, and shared memory allocations.
5. Injected Constant Buffers (CB0 for 64-bit kernel arguments, CB2 for compiler constants) directly into the GPU virtual address space.

### Phase 2: Q8_0 (W8A32) Quantization & Hand-Crafted Kepler CUBINs
Kepler GK107 lacks INT8 Tensor math. Running FP32 weights would exceed the 2 GB VRAM limit (TinyLlama FP32 is 4.4 GB).
- We designed a **Q8_0 (W8A32) quantization format**:
  - Blocks of 32 weights stored as 32 signed 8-bit integers (`int8_t`) with one 16-bit half-precision (`fp16`) scale factor.
  - Memory footprint: **34 bytes per 32 weights** (1.0625 bytes/weight, $3.76\times$ compression over FP32).
  - Dot-product kernel: Dequantizes on-the-fly inside the 65,536-register file and accumulates in FP32.
- Compiled native Kepler machine code directly to **CUBIN (ELF container)** targeting `sm_30`.

### Phase 3: The Complete Transformer Layer in Physical Silicon
We chained all 11 stages of a full Transformer decoder layer:
1. `rms_norm_kernel` (Input RMSNorm)
2. `gemv_q8_0_kernel` ($W_q, W_k, W_v$ projections)
3. `rope_kcache_kernel` (Rotary positional embeddings & KV-cache insertion)
4. `attention_decode_kernel` (Multi-head attention & KV reduction)
5. `gemv_q8_0_kernel` ($W_o$ output projection)
6. `residual_add_kernel` (Attention residual)
7. `rms_norm_kernel` (Post-attention RMSNorm)
8. `gemv_q8_0_kernel` ($W_{gate}$ projection)
9. `gemv_q8_0_kernel` ($W_{up}$ projection)
10. `swiglu_kernel` (SwiGLU non-linear activation: $\text{Swish}(x_{gate}) \odot x_{up}$)
11. `gemv_q8_0_kernel` ($W_{down}$ projection) + residual accumulation.

**Empirical Silicon Validation:**
- Numerical Cosine Similarity vs 64-bit CPU reference: **`1.00000000`**
- Relative $L_2$ Error: **$5.91 \times 10^{-5}$** (0 NaNs, 0 Infs).
- Per-layer latency: **`22.38 ms`** ($+23.7\%$ faster than FP32).

### Phase 4: Full 22-Layer TinyLlama-1.1B Autoregressive Runner
We exported all 22 layers of the official `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model (`model.safetensors`, 2.20 GB):
- 22 Decoder Layers: `layer_00.bin` .. `layer_21.bin` (47.27 MB each in VRAM).
- LM-Head: `lm_head.bin` (70.31 MB Q8_0 in VRAM).
- Final Norm: `final_norm.bin` (8 KB FP32 in VRAM).
- Host Embeddings: `embed_tokens.bin` (250 MB FP32 on CPU host).
- Unified Multilayer KV-Cache: **44.0 MB** in VRAM ($T_{\max}=1024$).
- **Total Resident VRAM:** **`1154.16 MB`** (~1.15 GB), leaving **~850 MB free** for Linux and the 2880x1800 Retina display.

### Phase 5: The P0 Turbo Boost Breakthrough & Dynamic DVFS
Initially, Nouveau locked Kepler at P5 idle clocks (405 MHz core, 1560 MHz GDDR5), yielding 1.2 tokens/second.
By unlocking write access to `/sys/kernel/debug/dri/0/pstate`, we unlocked state **`0f`**:
- **Core Clock:** $405\text{ MHz} \longrightarrow \mathbf{925\text{ MHz}}$ ($+128\%$)
- **GDDR5 Clock:** $1560\text{ MHz} \longrightarrow \mathbf{5016\text{ MHz}}$ ($+221\%$)
- **Autoregressive Generation:** Accelerated from **`1.20 t/s`** to **`3.93 tokens/second`** ($3.27\times$ speedup)!
- **Software DVFS Governor:** Implemented automatic step-up to `0f` on prompt arrival and immediate step-down to `0a` on completion for silent, passive thermal dissipation.

---

## 📊 Measured Benchmark Telemetry (Physical Hardware)

All tests performed on physical silicon: **NVIDIA GeForce GT 750M (Kepler GK107 sm_30, 2 GB GDDR5)** under **Linux 6.8.0-52-generic x86_64**.

### 1. TinyLlama-1.1B Generation Benchmarks
```text
[*] Prompt (76 tokens): Escribe exclusivamente un programa en lenguaje C con #include <stdio.h> para calcular el factorial...
[*] Starting Prefill Token-by-Token on Kepler Silicon (P0 Turbo 925 MHz)...
[+] Prefill Complete: 19928.37 ms (3.8 tokens/sec)

--- Assistant Response ---
#include <stdio.h>

int main() {
    int num, factorial = 1;
    printf("Enter a number: ");
    scanf("%d", &num);
    while (num > 1) {
        factorial *= num;
        num--;
    }
    printf("The factorial of %d is: %d\n", num, factorial);
    return 0;
}
--------------------------
[+] Generation Finished: 160 tokens in 45.87 s (3.49 tokens/sec)
[*] GPU Stepped Down to Idle (0a) for fast thermal dissipation.
```

### 2. Dual Silicon Comparison Matrix: GPU vs CPU

| Model | Parameters | Hardware Target | Backend / Driver | VRAM / RAM | Latency / Token | Throughput (t/s) | TTFT (Prefill) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B** | 0.5B | **GPU (GT 750M P0)** | OpenCL 3.0 (Rusticl) | $420\text{ MB VRAM}$ | **$222.8\text{ ms}$** | **$4.49\text{ t/s}$** ⚡ | $17.8\text{ s}$ |
| **Qwen2.5-0.5B** | 0.5B | **CPU (i7 Haswell)** | AVX2+FMA3 (8 Threads) | $\approx 600\text{ MB RAM}$ | **$227.2\text{ ms}$** | **$4.40\text{ t/s}$** | $5.3\text{ s}$ |
| **TinyLlama-1.1B** | 1.1B | **GPU (GT 750M P0)** | Native C + CUBIN sm_30 | $1150\text{ MB VRAM}$ | **$254.4\text{ ms}$** | **$3.93\text{ t/s}$** 🚀 | $11.2\text{ s}$ |
| **Qwen2.5-Coder-1.5B** | 1.5B | **GPU (GT 750M P0)** | OpenCL 3.0 (Rusticl) | $1110\text{ MB VRAM}$ | **$478.3\text{ ms}$** | **$2.09\text{ t/s}$** | $44.4\text{ s}$ |
| **Qwen2.5-Coder-1.5B** | 1.5B | **CPU (i7 Haswell)** | AVX2+FMA3 (8 Threads) | $\approx 1.5\text{ GB RAM}$ | **$226.7\text{ ms}$** | **$4.41\text{ t/s}$** ⚡ | $6.4\text{ s}$ |
| **Qwen2.5-Coder-3B** | 3.1B | **CPU (i7 Haswell)** | AVX2+FMA3 (8 Threads) | $\approx 2.8\text{ GB RAM}$ | **$778.8\text{ ms}$** | **$1.28\text{ t/s}$** 🧠 | $12.3\text{ s}$ |
| **DeepSeek-R1-8B** | 8.2B | **CPU (i7 Haswell)** | AVX2 (Thinking Mode) | $\approx 6.8\text{ GB RAM}$ | **$2693.5\text{ ms}$** | **$0.37\text{ t/s}$** 🧬 | $8.2\text{ s}$ |
| **Qwen2.5-Coder-7B** | 7.6B | **CPU (i7 Haswell)** | AVX2+FMA3 (8 Threads) | $\approx 5.5\text{ GB RAM}$ | **$2885.1\text{ ms}$** | **$0.35\text{ t/s}$** 🏰 | $26.0\text{ s}$ |

---

## 🛠️ Quick Start & Usage

### 1. Compile the Native Driver Shim & Runner
```bash
# Clone the repository
git clone https://github.com/fbetancourt-dev/kepler-cuda-native-llm.git
cd kepler-cuda-native-llm

# Build the CUDA shim library and the LLM runner
make all
```

### 2. Export TinyLlama Weights to Q8_0
```bash
# Download and export official weights to models/tinyllama_q8_0/
python3 tools/export_tinyllama_q8_0.py
```

### 3. Run Automated Silicon Regression Suites (20 / 20 PASS)
```bash
./run_all_silicon_suites.sh
```

### 4. Interactive Chat & Direct Prompts
```bash
# Direct one-shot prompt with custom token limit:
./bin/tinyllama_runner -n 250 "Write a C function to reverse an array in-place"

# Interactive Chat REPL:
./bin/tinyllama_runner
```
In interactive mode:
- Type your prompt and press Enter.
- Use `/tokens <N>` to change generation length dynamically.
- Use `/temp <F>` to adjust sampling temperature (0.0 for deterministic greedy).
- Type `exit` or press `Ctrl+C` to quit.

---

## 📜 Full 20-Suite Silicon Verification Matrix

| Suite # | Test Target | Description | Silicon Status |
| :---: | :--- | :--- | :---: |
| **#1** | `test_nouveau_raw` | DRM Channel Allocation & PFIFO subchannel mapping | **PASS (100%)** |
| **#2** | `test_compute_gemm` | 3D Compute Engine QMD dispatch | **PASS (100%)** |
| **#3** | `test_gemm_official` | General Matrix Multiplication (GEMM) | **PASS (100%)** |
| **#4** | `test_vector_add` | Vector elementwise accumulation | **PASS (100%)** |
| **#5** | `test_cubin_loader` | ELF CUBIN container parsing & symbol resolution | **PASS (100%)** |
| **#6** | `test_cuda_shim` | `cuModuleLoad`, `cuLaunchKernel`, `cuMemcpyDtoH` | **PASS (100%)** |
| **#7** | `test_constant_cb2` | Constant Buffer CB2 injection | **PASS (100%)** |
| **#8** | `test_memory_barrier` | Memory barriers & synchronization primitives | **PASS (100%)** |
| **#9** | `test_q8_0_quant` | Q8_0 format pack/unpack verification | **PASS (100%)** |
| **#10** | `test_gemv_q8_0` | Q8_0 Matrix-Vector dot product kernel | **PASS (100%)** |
| **#11** | `test_rms_norm` | Root Mean Square Normalization | **PASS (100%)** |
| **#12** | `test_rope` | Rotary Position Embedding computation | **PASS (100%)** |
| **#13** | `test_attention_decode` | Autoregressive Attention with KV-Cache | **PASS (100%)** |
| **#14** | `test_swiglu` | SwiGLU non-linear FFN activation | **PASS (100%)** |
| **#15** | `test_tokenizer` | Pure C SentencePiece BPE tokenizer | **PASS (100%)** |
| **#16** | `test_kcache_multiturn` | Multi-turn KV-Cache continuity & append | **PASS (100%)** |
| **#17** | `test_lm_head_q8_0` | LM-Head 32,000-logit projection in VRAM | **PASS (100%)** |
| **#18** | `test_residual_add` | In-place residual connection accumulation | **PASS (100%)** |
| **#19** | `test_transformer_layer_q8_0` | Unified 11-stage Transformer Layer in VRAM | **PASS (100%)** |
| **#20** | `test_tinyllama_runner` | End-to-end 22-Layer LLM Autoregressive Generation | **PASS (100%)** |

---

## 👥 Acknowledgments & Engineering Credits

- **Silicon:** NVIDIA Kepler GK107 (Designed in Santa Clara, 2012–2013).
- **Workstation:** Apple MacBook Pro Retina 15" Mid-2014 (MacBookPro11,3).
- **Core Architecture & Implementation:** Francisco Betancourt (`@fbetancourt-dev`) & Samantha (Sam / Google Antigravity).
- **Senior Peer Review Committee:** Google Gemini Pro, Anthropic Claude 3.5 Sonnet, DeepSeek R1, OpenAI Codex CLI.

---

*Licensed under the Apache License, Version 2.0. Dedicated to the hackers, pirates, and hardware enthusiasts who refuse to let great silicon die.* 🏴‍☠️
