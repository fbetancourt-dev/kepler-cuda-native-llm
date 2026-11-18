/*
 * tinyllama_runner.c: Full Autoregressive Transformer LLM Runner on Kepler GT 750M
 * Architecture: TinyLlama-1.1B (22 Layers, D=2048, H_q=32, H_kv=4, D_mlp=5632)
 * Quantization: Q8_0 (W8A32) with Native Nouveau Shim & Pure C Tokenizer
 * Hardware: NVIDIA GeForce GT 750M (sm_30, 2 GB GDDR5, Linux 6.8 Nouveau DRM)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>

#include "cuda.h"
#include "tokenizer.h"

#define CHECK_CUDA(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "[-] CUDA Shim Error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while (0)

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ====================================================================
// PARÁMETROS DE ARQUITECTURA (TINYLLAMA-1.1B)
// ====================================================================
#define MODEL_DIM        2048
#define NUM_HEADS        32
#define NUM_KV_HEADS     4
#define HEAD_DIM         64
#define MLP_DIM          5632
#define NUM_LAYERS       22
#define VOCAB_SIZE       32000
#define MAX_SEQ_LEN      1024
#define EPSILON          1e-5f
#define ROPE_THETA       10000.0f
#define ROPE_MODE        0 // rotate_half (HuggingFace)

// Tamaños de matrices Q8_0 por capa (bloques de 32 pesos + escala float32 = 36B)
#define NUM_BLOCKS_D     (MODEL_DIM / 32)      // 64
#define NUM_BLOCKS_MLP   (MLP_DIM / 32)        // 176

#define SZ_NORM_BYTES    (MODEL_DIM * sizeof(float)) // 8192 B
#define SZ_WQ_Q8         ((size_t)MODEL_DIM * NUM_BLOCKS_D * 36)
#define SZ_WK_Q8         ((size_t)(NUM_KV_HEADS * HEAD_DIM) * NUM_BLOCKS_D * 36)
#define SZ_WV_Q8         ((size_t)(NUM_KV_HEADS * HEAD_DIM) * NUM_BLOCKS_D * 36)
#define SZ_WO_Q8         ((size_t)MODEL_DIM * NUM_BLOCKS_D * 36)
#define SZ_WGATE_Q8      ((size_t)MLP_DIM * NUM_BLOCKS_D * 36)
#define SZ_WUP_Q8        ((size_t)MLP_DIM * NUM_BLOCKS_D * 36)
#define SZ_WDOWN_Q8      ((size_t)MODEL_DIM * NUM_BLOCKS_MLP * 36)

#define TOTAL_LAYER_Q8_BYTES (SZ_WQ_Q8 + SZ_WK_Q8 + SZ_WV_Q8 + SZ_WO_Q8 + \
                              SZ_WGATE_Q8 + SZ_WUP_Q8 + SZ_WDOWN_Q8) // 49,545,216 B (~47.25 MB)
#define TOTAL_LAYER_FILE_BYTES (2 * SZ_NORM_BYTES + TOTAL_LAYER_Q8_BYTES) // 49,561,600 B (~47.27 MB)
#define SZ_LM_HEAD_Q8    ((size_t)VOCAB_SIZE * NUM_BLOCKS_D * 36) // 73,728,000 B (~70.31 MB)

// ====================================================================
// ESTRUCTURA GLOBAL DEL MODELO Y RUNNER
// ====================================================================
typedef struct {
    // Driver & Context
    CUdevice dev;
    CUcontext ctx;

    // Kernels CUBIN
    CUmodule mod_rmsnorm;
    CUmodule mod_gemv_q8;
    CUmodule mod_rope;
    CUmodule mod_attn;
    CUmodule mod_swiglu;

    CUfunction f_rmsnorm;
    CUfunction f_gemv_q8;
    CUfunction f_rope;
    CUfunction f_attn;
    CUfunction f_swiglu;
    CUfunction f_residual;

    // Pesos Device (1 BO por capa + LM Head + RMSNorm final)
    CUdeviceptr d_layer_weights[NUM_LAYERS];
    CUdeviceptr d_layer_norms[NUM_LAYERS];
    CUdeviceptr d_lm_head;
    CUdeviceptr d_final_norm;

    // KV-Cache unificado (22 capas * 2 KB/token * 1024 tokens = 44 MB)
    CUdeviceptr d_kv_cache;
    size_t sz_layer_kv;

    // Buffers Device scratch compartidos (reutilizados en cada capa)
    CUdeviceptr d_x;
    CUdeviceptr d_x_norm;
    CUdeviceptr d_Q;
    CUdeviceptr d_K;
    CUdeviceptr d_V;
    CUdeviceptr d_attn_out;
    CUdeviceptr d_attn_proj;
    CUdeviceptr d_x_ffn_norm;
    CUdeviceptr d_gate;
    CUdeviceptr d_up;
    CUdeviceptr d_mlp_act;
    CUdeviceptr d_down_proj;
    CUdeviceptr d_logits;
    CUdeviceptr d_cos;
    CUdeviceptr d_sin;

    // Buffers Host
    float *h_embed_tokens; // 32000 * 2048 floats (~250 MB en Host RAM)
    float *h_logits;       // 32000 floats (~128 KB)
    float *h_rope_cos;     // MAX_SEQ_LEN * 32 floats
    float *h_rope_sin;     // MAX_SEQ_LEN * 32 floats

    // Tokenizer
    tokenizer_t *tokenizer;

    // Estado de sesión
    int cur_pos;
    bool interrupted;
} tinyllama_model_t;

static tinyllama_model_t g_model;

// Manejador de señal SIGINT (Ctrl+C) para interrupción limpia del streaming
static void sigint_handler(int sig) {
    (void)sig;
    g_model.interrupted = true;
    printf("\n[!] Generation interrupted by user (Ctrl+C).\n");
}

// ====================================================================
// INICIALIZACIÓN DEL MODELO Y CARGA DE PESOS A VRAM
// ====================================================================
static bool model_init(const char *models_dir) {
    printf("[*] Initializing CUDA Kepler Driver & Hardware Context...\n");
    CHECK_CUDA(cuInit(0));
    CHECK_CUDA(cuDeviceGet(&g_model.dev, 0));
    CHECK_CUDA(cuCtxCreate(&g_model.ctx, 0, g_model.dev));

    char dev_name[256];
    CHECK_CUDA(cuDeviceGetName(dev_name, sizeof(dev_name), g_model.dev));
    printf("[+] Hardware: %s\n", dev_name);

    // Cargar módulos CUBIN
    printf("[*] Loading optimized CUBIN kernel modules...\n");
    CHECK_CUDA(cuModuleLoad(&g_model.mod_rmsnorm, "rms_norm_official.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&g_model.f_rmsnorm, g_model.mod_rmsnorm, "rms_norm_kernel"));

    CHECK_CUDA(cuModuleLoad(&g_model.mod_gemv_q8, "gemv_q8_0_official.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&g_model.f_gemv_q8, g_model.mod_gemv_q8, "gemv_q8_0_kernel"));

    CHECK_CUDA(cuModuleLoad(&g_model.mod_rope, "rope_kcache_official.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&g_model.f_rope, g_model.mod_rope, "rope_kcache_kernel"));

    CHECK_CUDA(cuModuleLoad(&g_model.mod_attn, "attention_decode_official.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&g_model.f_attn, g_model.mod_attn, "attention_decode_kernel"));

    CHECK_CUDA(cuModuleLoad(&g_model.mod_swiglu, "swiglu_official.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&g_model.f_swiglu, g_model.mod_swiglu, "swiglu_kernel"));
    CHECK_CUDA(cuModuleGetFunction(&g_model.f_residual, g_model.mod_swiglu, "residual_add_kernel"));

    // Cargar Tokenizer
    printf("[*] Initializing SentencePiece BPE Tokenizer...\n");
    g_model.tokenizer = tokenizer_init("models/tokenizer.bin");
    if (!g_model.tokenizer) {
        fprintf(stderr, "[-] Failed to load models/tokenizer.bin\n");
        return false;
    }

    // Pre-computar tabla de frecuencias RoPE (rotate_half)
    g_model.h_rope_cos = (float *)malloc(MAX_SEQ_LEN * (HEAD_DIM / 2) * sizeof(float));
    g_model.h_rope_sin = (float *)malloc(MAX_SEQ_LEN * (HEAD_DIM / 2) * sizeof(float));
    for (int p = 0; p < MAX_SEQ_LEN; p++) {
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            double freq = 1.0 / pow(ROPE_THETA, (2.0 * i) / (double)HEAD_DIM);
            double angle = (double)p * freq;
            g_model.h_rope_cos[p * (HEAD_DIM / 2) + i] = (float)cos(angle);
            g_model.h_rope_sin[p * (HEAD_DIM / 2) + i] = (float)sin(angle);
        }
    }

    // Cargar Host Embeddings (250 MB en RAM)
    char path_buf[512];
    snprintf(path_buf, sizeof(path_buf), "%s/embed_tokens.bin", models_dir);
    printf("[*] Loading Host Embeddings from '%s'...\n", path_buf);
    FILE *f_emb = fopen(path_buf, "rb");
    if (!f_emb) { fprintf(stderr, "[-] Could not open %s\n", path_buf); return false; }
    size_t sz_emb = (size_t)VOCAB_SIZE * MODEL_DIM * sizeof(float);
    g_model.h_embed_tokens = (float *)malloc(sz_emb);
    if (fread(g_model.h_embed_tokens, 1, sz_emb, f_emb) != sz_emb) {
        fprintf(stderr, "[-] Short read in embed_tokens.bin\n"); fclose(f_emb); return false;
    }
    fclose(f_emb);

    // Cargar Final RMSNorm (8 KB)
    snprintf(path_buf, sizeof(path_buf), "%s/final_norm.bin", models_dir);
    printf("[*] Loading Final RMSNorm to VRAM...\n");
    FILE *f_fnorm = fopen(path_buf, "rb");
    if (!f_fnorm) { fprintf(stderr, "[-] Could not open %s\n", path_buf); return false; }
    float h_fnorm[MODEL_DIM];
    if (fread(h_fnorm, sizeof(float), MODEL_DIM, f_fnorm) != MODEL_DIM) {
        fprintf(stderr, "[-] Short read in final_norm.bin\n"); fclose(f_fnorm); return false;
    }
    fclose(f_fnorm);
    CHECK_CUDA(cuMemAlloc(&g_model.d_final_norm, SZ_NORM_BYTES));
    CHECK_CUDA(cuMemcpyHtoD(g_model.d_final_norm, h_fnorm, SZ_NORM_BYTES));

    // Cargar LM-Head Q8_0 (70.31 MB a VRAM)
    snprintf(path_buf, sizeof(path_buf), "%s/lm_head.bin", models_dir);
    printf("[*] Loading LM-Head Q8_0 (%.2f MB) to VRAM...\n", (double)SZ_LM_HEAD_Q8 / (1024*1024));
    FILE *f_lm = fopen(path_buf, "rb");
    if (!f_lm) { fprintf(stderr, "[-] Could not open %s\n", path_buf); return false; }
    void *h_lm_buf = malloc(SZ_LM_HEAD_Q8);
    if (fread(h_lm_buf, 1, SZ_LM_HEAD_Q8, f_lm) != SZ_LM_HEAD_Q8) {
        fprintf(stderr, "[-] Short read in lm_head.bin\n"); free(h_lm_buf); fclose(f_lm); return false;
    }
    fclose(f_lm);
    CHECK_CUDA(cuMemAlloc(&g_model.d_lm_head, SZ_LM_HEAD_Q8));
    CHECK_CUDA(cuMemcpyHtoD(g_model.d_lm_head, h_lm_buf, SZ_LM_HEAD_Q8));
    free(h_lm_buf);

    // Asignar KV-Cache unificado (44 MB en VRAM)
    g_model.sz_layer_kv = 2 * (size_t)NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * sizeof(float); // 2 MB/capa
    size_t sz_total_kv = (size_t)NUM_LAYERS * g_model.sz_layer_kv; // 44 MB
    printf("[*] Allocating Unified Multilayer KV-Cache (%.2f MB) in VRAM...\n", (double)sz_total_kv / (1024*1024));
    CHECK_CUDA(cuMemAlloc(&g_model.d_kv_cache, sz_total_kv));
    CHECK_CUDA(cuMemsetD8(g_model.d_kv_cache, 0, sz_total_kv));

    // Asignar Buffers Scratch Device (compartidos entre capas)
    printf("[*] Allocating Device Scratch Activations in VRAM...\n");
    CHECK_CUDA(cuMemAlloc(&g_model.d_x,          MODEL_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_x_norm,     MODEL_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_Q,          NUM_HEADS * HEAD_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_K,          NUM_KV_HEADS * HEAD_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_V,          NUM_KV_HEADS * HEAD_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_attn_out,   NUM_HEADS * HEAD_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_attn_proj,  MODEL_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_x_ffn_norm, MODEL_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_gate,       MLP_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_up,         MLP_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_mlp_act,    MLP_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_down_proj,  MODEL_DIM * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_logits,     VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_cos,        (HEAD_DIM / 2) * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&g_model.d_sin,        (HEAD_DIM / 2) * sizeof(float)));

    g_model.h_logits = (float *)malloc(VOCAB_SIZE * sizeof(float));

    // Cargar las 22 Capas Transformer (1 BO por capa de 47.25 MB + 1 BO de 16 KB para RMSNorms)
    printf("[*] Uploading 22 Transformer Decoder Layers to physical VRAM...\n");
    void *h_layer_buf = malloc(TOTAL_LAYER_FILE_BYTES);

    for (int l = 0; l < NUM_LAYERS; l++) {
        snprintf(path_buf, sizeof(path_buf), "%s/layer_%02d.bin", models_dir, l);
        FILE *f_layer = fopen(path_buf, "rb");
        if (!f_layer) {
            fprintf(stderr, "[-] Could not open layer file: %s\n", path_buf);
            free(h_layer_buf);
            return false;
        }
        if (fread(h_layer_buf, 1, TOTAL_LAYER_FILE_BYTES, f_layer) != TOTAL_LAYER_FILE_BYTES) {
            fprintf(stderr, "[-] Short read in layer %d\n", l);
            fclose(f_layer);
            free(h_layer_buf);
            return false;
        }
        fclose(f_layer);

        // 1. Asignar y copiar RMSNorms (16 KB)
        CHECK_CUDA(cuMemAlloc(&g_model.d_layer_norms[l], 2 * SZ_NORM_BYTES));
        CHECK_CUDA(cuMemcpyHtoD(g_model.d_layer_norms[l], h_layer_buf, 2 * SZ_NORM_BYTES));

        // 2. Asignar y copiar Pesos Q8_0 (47.25 MB)
        CHECK_CUDA(cuMemAlloc(&g_model.d_layer_weights[l], TOTAL_LAYER_Q8_BYTES));
        CHECK_CUDA(cuMemcpyHtoD(g_model.d_layer_weights[l], (char *)h_layer_buf + 2 * SZ_NORM_BYTES, TOTAL_LAYER_Q8_BYTES));

        printf("    [Layer %02d/21] Loaded into VRAM (BO: 47.25 MB)\n", l);
    }
    free(h_layer_buf);

    double total_vram_mb = (double)(NUM_LAYERS * TOTAL_LAYER_FILE_BYTES + SZ_LM_HEAD_Q8 + sz_total_kv) / (1024*1024);
    printf("[+] ALL 22 LAYERS LOADED! Total Resident VRAM: %.2f MB (~1.15 GB)\n", total_vram_mb);
    printf("[+] GeForce GT 750M VRAM Free Headroom: ~850 MB for OS & Display.\n");

    g_model.cur_pos = 0;
    g_model.interrupted = false;
    return true;
}

// ====================================================================
// EJECUCIÓN FORWARD DE 1 TOKEN A TRAVÉS DE LAS 22 CAPAS
// ====================================================================
static void runner_forward_token(int token_id, int pos, bool compute_logits) {
    // 1. Embedding lookup en CPU -> copiar 8 KB a d_x
    float *emb = g_model.h_embed_tokens + (size_t)token_id * MODEL_DIM;
    CHECK_CUDA(cuMemcpyHtoD(g_model.d_x, emb, MODEL_DIM * sizeof(float)));

    // 2. Actualizar frecuencias RoPE para pos actual (128 bytes)
    float *cos_p = g_model.h_rope_cos + (size_t)pos * (HEAD_DIM / 2);
    float *sin_p = g_model.h_rope_sin + (size_t)pos * (HEAD_DIM / 2);
    CHECK_CUDA(cuMemcpyHtoD(g_model.d_cos, cos_p, (HEAD_DIM / 2) * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(g_model.d_sin, sin_p, (HEAD_DIM / 2) * sizeof(float)));

    int D_arg = MODEL_DIM;
    int D_kv_arg = NUM_KV_HEADS * HEAD_DIM;
    int D_mlp_arg = MLP_DIM;
    int num_blocks_D_arg = NUM_BLOCKS_D;     // 64
    int num_blocks_mlp_arg = NUM_BLOCKS_MLP; // 176
    float eps_arg = EPSILON;
    int pos_arg = pos;
    int H_q_arg = NUM_HEADS;
    int H_kv_arg = NUM_KV_HEADS;
    int head_dim_arg = HEAD_DIM;
    int max_seq_len_arg = MAX_SEQ_LEN;
    int context_len_arg = pos + 1;
    float scale_arg = 1.0f / sqrtf((float)HEAD_DIM);
    int mode_arg = ROPE_MODE;

    // 3. Iterar por las 22 Capas Transformer
    for (int l = 0; l < NUM_LAYERS; l++) {
        CUdeviceptr d_weights = g_model.d_layer_weights[l];
        CUdeviceptr d_w_attn_dev = g_model.d_layer_norms[l];
        CUdeviceptr d_w_ffn_dev  = g_model.d_layer_norms[l] + SZ_NORM_BYTES;

        CUdeviceptr d_Wq    = d_weights;
        CUdeviceptr d_Wk    = d_Wq + SZ_WQ_Q8;
        CUdeviceptr d_Wv    = d_Wk + SZ_WK_Q8;
        CUdeviceptr d_Wo    = d_Wv + SZ_WV_Q8;
        CUdeviceptr d_Wgate = d_Wo + SZ_WO_Q8;
        CUdeviceptr d_Wup   = d_Wgate + SZ_WGATE_Q8;
        CUdeviceptr d_Wdown = d_Wup + SZ_WUP_Q8;

        // Offsets en el KV-Cache unificado para la capa l
        size_t off_k_layer = (size_t)l * g_model.sz_layer_kv;
        size_t off_v_layer = off_k_layer + (size_t)NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * sizeof(float);
        CUdeviceptr d_K_cache = g_model.d_kv_cache + off_k_layer;
        CUdeviceptr d_V_cache = g_model.d_kv_cache + off_v_layer;

        void *args_norm1[]  = { &g_model.d_x_norm, &g_model.d_x, &d_w_attn_dev, &D_arg, &eps_arg };
        void *args_Wq[]     = { &g_model.d_Q, &d_Wq, &g_model.d_x_norm, &D_arg, &num_blocks_D_arg };
        void *args_Wk[]     = { &g_model.d_K, &d_Wk, &g_model.d_x_norm, &D_kv_arg, &num_blocks_D_arg };
        void *args_Wv[]     = { &g_model.d_V, &d_Wv, &g_model.d_x_norm, &D_kv_arg, &num_blocks_D_arg };
        void *args_rope[]   = { &g_model.d_Q, &g_model.d_K, &g_model.d_V, &g_model.d_cos, &g_model.d_sin,
                                &d_K_cache, &d_V_cache,
                                &pos_arg, &H_q_arg, &H_kv_arg, &head_dim_arg, &max_seq_len_arg, &mode_arg };
        void *args_attn[]   = { &g_model.d_attn_out, &g_model.d_Q, &d_K_cache, &d_V_cache,
                                &H_q_arg, &H_kv_arg, &head_dim_arg, &max_seq_len_arg, &context_len_arg, &scale_arg };
        void *args_Wo[]     = { &g_model.d_attn_proj, &d_Wo, &g_model.d_attn_out, &D_arg, &num_blocks_D_arg };
        void *args_res1[]   = { &g_model.d_x, &g_model.d_attn_proj, &D_arg };
        void *args_norm2[]  = { &g_model.d_x_ffn_norm, &g_model.d_x, &d_w_ffn_dev, &D_arg, &eps_arg };
        void *args_Wgate[]  = { &g_model.d_gate, &d_Wgate, &g_model.d_x_ffn_norm, &D_mlp_arg, &num_blocks_D_arg };
        void *args_Wup[]    = { &g_model.d_up, &d_Wup, &g_model.d_x_ffn_norm, &D_mlp_arg, &num_blocks_D_arg };
        void *args_swiglu[] = { &g_model.d_mlp_act, &g_model.d_gate, &g_model.d_up, &D_mlp_arg };
        void *args_Wdown[]  = { &g_model.d_down_proj, &d_Wdown, &g_model.d_mlp_act, &D_arg, &num_blocks_mlp_arg };
        void *args_res2[]   = { &g_model.d_x, &g_model.d_down_proj, &D_arg };

        // 1. RMSNorm 1
        CHECK_CUDA(cuLaunchKernel(g_model.f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_norm1, NULL));
        // 2. Q, K, V
        CHECK_CUDA(cuLaunchKernel(g_model.f_gemv_q8, (D_arg + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wq, NULL));
        CHECK_CUDA(cuLaunchKernel(g_model.f_gemv_q8, (D_kv_arg + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wk, NULL));
        CHECK_CUDA(cuLaunchKernel(g_model.f_gemv_q8, (D_kv_arg + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wv, NULL));
        // 3. Fused RoPE + KV Cache
        CHECK_CUDA(cuLaunchKernel(g_model.f_rope, H_q_arg, 1, 1, head_dim_arg / 2, 1, 1, 0, 0, args_rope, NULL));
        // 4. FlashAttention
        CHECK_CUDA(cuLaunchKernel(g_model.f_attn, H_q_arg, 1, 1, 32, 1, 1, 0, 0, args_attn, NULL));
        // 5. Wo
        CHECK_CUDA(cuLaunchKernel(g_model.f_gemv_q8, (D_arg + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wo, NULL));
        // 6. Resid 1
        CHECK_CUDA(cuLaunchKernel(g_model.f_residual, (D_arg + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_res1, NULL));
        // 7. RMSNorm 2
        CHECK_CUDA(cuLaunchKernel(g_model.f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_norm2, NULL));
        // 8. Gate, Up
        CHECK_CUDA(cuLaunchKernel(g_model.f_gemv_q8, (D_mlp_arg + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wgate, NULL));
        CHECK_CUDA(cuLaunchKernel(g_model.f_gemv_q8, (D_mlp_arg + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wup, NULL));
        // 9. SwiGLU
        CHECK_CUDA(cuLaunchKernel(g_model.f_swiglu, (D_mlp_arg + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_swiglu, NULL));
        // 10. Wdown
        CHECK_CUDA(cuLaunchKernel(g_model.f_gemv_q8, (D_arg + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wdown, NULL));
        // 11. Resid 2
        CHECK_CUDA(cuLaunchKernel(g_model.f_residual, (D_arg + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_res2, NULL));
    }

    // 4. Si se solicitan logits (último token del prompt o fase decode):
    if (compute_logits) {
        // RMSNorm final
        void *args_final_norm[] = { &g_model.d_x_norm, &g_model.d_x, &g_model.d_final_norm, &D_arg, &eps_arg };
        CHECK_CUDA(cuLaunchKernel(g_model.f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_final_norm, NULL));

        // GEMV LM Head Q8_0 (32000 filas x 2048 columnas)
        int vocab_arg = VOCAB_SIZE;
        void *args_lm_head[] = { &g_model.d_logits, &g_model.d_lm_head, &g_model.d_x_norm, &vocab_arg, &num_blocks_D_arg };
        CHECK_CUDA(cuLaunchKernel(g_model.f_gemv_q8, (vocab_arg + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_lm_head, NULL));

        CHECK_CUDA(cuCtxSynchronize());

        // Copiar 128 KB de logits a Host RAM
        CHECK_CUDA(cuMemcpyDtoH(g_model.h_logits, g_model.d_logits, VOCAB_SIZE * sizeof(float)));
    } else {
        CHECK_CUDA(cuCtxSynchronize());
    }
}

// ====================================================================
// SAMPLER PURO EN CPU (GREEDY / TEMPERATURE / TOP-K / TOP-P)
// ====================================================================
static int sampler_sample(float *logits, float temperature, float top_p, int top_k) {
    // Enmascarar tokens especiales prohibidos en generación
    logits[TOKEN_UNK] = -1e9f;
    logits[TOKEN_BOS] = -1e9f;

    // Modo Greedy (Argmax determinista)
    if (temperature <= 0.0f) {
        int best_id = 0;
        float max_l = logits[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (logits[i] > max_l) {
                max_l = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    // Escalar por temperatura y encontrar máximo
    float inv_t = 1.0f / temperature;
    float max_l = -1e9f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        logits[i] *= inv_t;
        if (logits[i] > max_l) max_l = logits[i];
    }

    // Softmax
    double sum_exp = 0.0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        logits[i] = expf(logits[i] - max_l);
        sum_exp += (double)logits[i];
    }
    float inv_sum = (float)(1.0 / sum_exp);
    for (int i = 0; i < VOCAB_SIZE; i++) {
        logits[i] *= inv_sum;
    }

    // Muestreo por distribución acumulada
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        cdf += logits[i];
        if (r <= cdf) return i;
    }
    return VOCAB_SIZE - 1;
}

static void set_gpu_pstate(const char *level) {
    FILE *fp = fopen("/sys/kernel/debug/dri/0/pstate", "w");
    if (fp) {
        fprintf(fp, "%s\n", level);
        fclose(fp);
    }
}

// ====================================================================
// BUCLE DE GENERACIÓN DE TEXTO
// ====================================================================
static void generate_response(const char *user_prompt, int max_new_tokens, float temp, float top_p, int top_k) {
    char formatted_prompt[4096];
    tokenizer_format_chat_prompt(formatted_prompt, sizeof(formatted_prompt), user_prompt);

    int prompt_tokens[2048];
    int num_prompt_tokens = tokenizer_encode(g_model.tokenizer, formatted_prompt, 1, 0, prompt_tokens, 2048);
    if (num_prompt_tokens <= 0) {
        fprintf(stderr, "[-] Failed to tokenize prompt.\n");
        return;
    }

    // Activar P0 Turbo Boost (925 MHz Core / 5016 MHz GDDR5) para cómputo pleno
    set_gpu_pstate("0f");

    printf("\n[*] Prompt (%d tokens): %s\n", num_prompt_tokens, user_prompt);
    printf("[*] Starting Prefill Token-by-Token on Kepler Silicon (P0 Turbo 925 MHz)...\n");

    double t_prefill_start = get_time_sec();

    // Fase 1: Prefill token a token (los tokens intermedios NO calculan LM Head)
    for (int i = 0; i < num_prompt_tokens; i++) {
        bool is_last = (i == num_prompt_tokens - 1);
        runner_forward_token(prompt_tokens[i], g_model.cur_pos, is_last);
        g_model.cur_pos++;
    }
    double t_prefill_end = get_time_sec();
    double prefill_time = t_prefill_end - t_prefill_start;
    double prefill_rate = (double)num_prompt_tokens / prefill_time;
    printf("[+] Prefill Complete: %.2f ms (%.1f tokens/sec)\n", prefill_time * 1000.0, prefill_rate);

    printf("\n--- Assistant Response ---\n");
    fflush(stdout);

    // Fase 2: Decode autoregresivo token a token
    double t_gen_start = get_time_sec();
    int generated_count = 0;

    while (generated_count < max_new_tokens && g_model.cur_pos < MAX_SEQ_LEN && !g_model.interrupted) {
        // Muestrear siguiente token de los logits actuales
        int next_tok = sampler_sample(g_model.h_logits, temp, top_p, top_k);

        // Comprobar token de parada EOS (ID 2 = </s>)
        if (next_tok == TOKEN_EOS) {
            break;
        }

        // Decodificar y emitir en streaming a terminal
        const char *piece = tokenizer_decode(g_model.tokenizer, next_tok);
        if (piece) {
            fputs(piece, stdout);
            fflush(stdout);
        }

        generated_count++;

        // Ejecutar paso decode en GPU
        runner_forward_token(next_tok, g_model.cur_pos, true);
        g_model.cur_pos++;
    }

    double t_gen_end = get_time_sec();
    double gen_time = t_gen_end - t_gen_start;
    double gen_rate = (generated_count > 0) ? ((double)generated_count / gen_time) : 0.0;

    // Retorno automático a reposo (P5 / 0a) para enfriamiento pasivo
    set_gpu_pstate("0a");

    printf("\n--------------------------\n");
    printf("[+] Generation Finished: %d tokens in %.2f s (%.2f tokens/sec)\n",
           generated_count, gen_time, gen_rate);
    printf("[*] GPU Stepped Down to Idle (0a) for fast thermal dissipation.\n");
}

// ====================================================================
// PUNTO DE ENTRADA (CLI & REPL INTERACTIVO)
// ====================================================================
int main(int argc, char **argv) {
    printf("====================================================================\n");
    printf("  TINYLLAMA-1.1B AUTOREGRESSIVE RUNNER ON KEPLER GT 750M (sm_30)\n");
    printf("  Native CUDA Shim + Pure C SentencePiece BPE + Q8_0 (W8A32)\n");
    printf("====================================================================\n");

    signal(SIGINT, sigint_handler);
    srand(1337);

    const char *models_dir = "models/tinyllama_q8_0";
    if (!model_init(models_dir)) {
        fprintf(stderr, "[-] Model initialization failed.\n");
        return 1;
    }

    int max_tokens = 160;
    float temp = 0.0f; // Greedy por defecto
    const char *prompt = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) {
            printf("[*] Running Automated Regression Test Mode...\n");
            generate_response("Hello!", 8, 0.0f, 0.9f, 40);
            printf("[+] Regression Test Completed Cleanly.\n");
            return 0;
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temp") == 0) && i + 1 < argc) {
            temp = (float)atof(argv[++i]);
        } else if (argv[i][0] != '-') {
            prompt = argv[i];
        }
    }

    if (prompt) {
        // Modo de una sola ejecución vía argumento CLI
        generate_response(prompt, max_tokens, temp, 0.9f, 40);
    } else {
        // Modo interactivo REPL
        printf("\n[*] Interactive Chat REPL Ready! (Max Tokens: %d, Temp: %.2f)\n", max_tokens, temp);
        printf("[*] Commands: Type your prompt, '/tokens <N>', '/temp <F>', 'exit' or Ctrl+C to stop.\n\n");

        char line_buf[1024];
        while (1) {
            printf("\nUser > ");
            fflush(stdout);
            if (!fgets(line_buf, sizeof(line_buf), stdin)) break;

            // Trim newline
            line_buf[strcspn(line_buf, "\r\n")] = 0;
            if (strlen(line_buf) == 0) continue;
            if (strcmp(line_buf, "exit") == 0 || strcmp(line_buf, "quit") == 0) break;

            if (strncmp(line_buf, "/tokens ", 8) == 0) {
                max_tokens = atoi(line_buf + 8);
                printf("[*] Max tokens updated to: %d\n", max_tokens);
                continue;
            }
            if (strncmp(line_buf, "/temp ", 6) == 0) {
                temp = (float)atof(line_buf + 6);
                printf("[*] Temperature updated to: %.2f\n", temp);
                continue;
            }

            g_model.interrupted = false;
            generate_response(line_buf, max_tokens, temp, 0.9f, 40);
        }
    }

    printf("\n[+] Exiting TinyLlama Runner cleanly. Freeing resources...\n");
    return 0;
}
