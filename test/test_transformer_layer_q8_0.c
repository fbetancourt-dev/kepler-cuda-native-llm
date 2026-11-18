#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "[-] CUDA Error in %s (line %d): code %d\n", #call, __LINE__, err); \
        return 1; \
    } \
} while (0)

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ====================================================================
// ESTRUCTURA Q8_0 (32 int8 + escala float32 = 36 bytes)
// ====================================================================

typedef struct {
    float d;
    int8_t qs[32];
} block_q8_0;

// Cuantizador en Host
static void quantize_row_q8_0(block_q8_0 *out_blocks, const float *in_row, int K) {
    int num_blocks = K / 32;
    for (int b = 0; b < num_blocks; b++) {
        const float *block_src = in_row + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float v = fabsf(block_src[i]);
            if (v > amax) amax = v;
        }

        float scale = amax / 127.0f;
        out_blocks[b].d = scale;

        float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;
        for (int i = 0; i < 32; i++) {
            int q = (int)roundf(block_src[i] * inv_scale);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out_blocks[b].qs[i] = (int8_t)q;
        }
    }
}

// ====================================================================
// REFERENCIA CPU PRECISA (DOUBLE PRECISION)
// ====================================================================

static void cpu_rms_norm(float *y, const float *x, const float *w, int dim, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        double val = (double)x[i];
        sum_sq += val * val;
    }
    double mean_sq = sum_sq / (double)dim;
    double rrms = 1.0 / sqrt(mean_sq + (double)eps);
    for (int i = 0; i < dim; i++) {
        y[i] = (float)((double)x[i] * rrms * (double)w[i]);
    }
}

static void cpu_gemv_q8_0(float *y, const block_q8_0 *A, const float *x, int M, int K) {
    int num_blocks = K / 32;
    for (int r = 0; r < M; r++) {
        const block_q8_0 *row_blocks = A + (size_t)r * num_blocks;
        double row_sum = 0.0;
        for (int b = 0; b < num_blocks; b++) {
            double block_sum = 0.0;
            const float *x_block = x + b * 32;
            for (int i = 0; i < 32; i++) {
                block_sum += (double)row_blocks[b].qs[i] * (double)x_block[i];
            }
            row_sum += block_sum * (double)row_blocks[b].d;
        }
        y[r] = (float)row_sum;
    }
}

static void cpu_rope_kcache(
    float *Q, const float *K_in, const float *V_in,
    const float *cos_pos, const float *sin_pos,
    float *K_cache, float *V_cache,
    int pos, int num_heads, int num_kv_heads, int head_dim, int max_seq_len, int mode)
{
    int half_dim = head_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *q_h = Q + (size_t)h * head_dim;
        if (mode == 0) { // rotate_half (HuggingFace oficial para TinyLlama)
            for (int i = 0; i < half_dim; i++) {
                double c = (double)cos_pos[i];
                double s = (double)sin_pos[i];
                double q0 = (double)q_h[i];
                double q1 = (double)q_h[i + half_dim];
                q_h[i]            = (float)(q0 * c - q1 * s);
                q_h[i + half_dim] = (float)(q0 * s + q1 * c);
            }
        } else { // interleaved (Meta)
            for (int i = 0; i < half_dim; i++) {
                double c = (double)cos_pos[i];
                double s = (double)sin_pos[i];
                double q0 = (double)q_h[i * 2];
                double q1 = (double)q_h[i * 2 + 1];
                q_h[i * 2]     = (float)(q0 * c - q1 * s);
                q_h[i * 2 + 1] = (float)(q0 * s + q1 * c);
            }
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        const float *k_h = K_in + (size_t)h * head_dim;
        const float *v_h = V_in + (size_t)h * head_dim;
        size_t off = ((size_t)h * max_seq_len + pos) * head_dim;
        if (mode == 0) {
            for (int i = 0; i < half_dim; i++) {
                double c = (double)cos_pos[i];
                double s = (double)sin_pos[i];
                double k0 = (double)k_h[i];
                double k1 = (double)k_h[i + half_dim];
                K_cache[off + i]            = (float)(k0 * c - k1 * s);
                K_cache[off + i + half_dim] = (float)(k0 * s + k1 * c);
                V_cache[off + i]            = v_h[i];
                V_cache[off + i + half_dim] = v_h[i + half_dim];
            }
        } else {
            for (int i = 0; i < half_dim; i++) {
                double c = (double)cos_pos[i];
                double s = (double)sin_pos[i];
                double k0 = (double)k_h[i * 2];
                double k1 = (double)k_h[i * 2 + 1];
                K_cache[off + i * 2]     = (float)(k0 * c - k1 * s);
                K_cache[off + i * 2 + 1] = (float)(k0 * s + k1 * c);
                V_cache[off + i * 2]     = v_h[i * 2];
                V_cache[off + i * 2 + 1] = v_h[i * 2 + 1];
            }
        }
    }
}

static void cpu_attention_decode(
    float *out, const float *Q, const float *K_cache, const float *V_cache,
    int num_heads, int num_kv_heads, int head_dim, int max_seq_len, int context_len, float scale)
{
    int gqa_ratio = num_heads / num_kv_heads;
    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / gqa_ratio;
        const float *q_h = Q + (size_t)h * head_dim;
        float *out_h = out + (size_t)h * head_dim;

        double max_score = -1e30;
        double sum_exp = 0.0;
        double acc[64] = {0.0};

        for (int t = 0; t < context_len; t++) {
            size_t off = ((size_t)kv_h * max_seq_len + t) * head_dim;
            const float *k_vec = K_cache + off;
            const float *v_vec = V_cache + off;

            double dot = 0.0;
            for (int i = 0; i < head_dim; i++) {
                dot += (double)q_h[i] * (double)k_vec[i];
            }
            double score = dot * (double)scale;

            double new_max = (score > max_score) ? score : max_score;
            double alpha = exp(max_score - new_max);
            double beta = exp(score - new_max);

            sum_exp = sum_exp * alpha + beta;
            for (int i = 0; i < head_dim; i++) {
                acc[i] = acc[i] * alpha + beta * (double)v_vec[i];
            }
            max_score = new_max;
        }

        double inv_sum = (sum_exp > 1e-12) ? (1.0 / sum_exp) : 0.0;
        for (int i = 0; i < head_dim; i++) {
            out_h[i] = (float)(acc[i] * inv_sum);
        }
    }
}

static void cpu_swiglu(float *out, const float *gate, const float *up, int dim) {
    for (int i = 0; i < dim; i++) {
        double g = (double)gate[i];
        double u = (double)up[i];
        double silu_g = g / (1.0 + exp(-g));
        out[i] = (float)(silu_g * u);
    }
}

static void cpu_residual_add(float *x, const float *delta, int dim) {
    for (int i = 0; i < dim; i++) {
        x[i] += delta[i];
    }
}

// ====================================================================
// TEST HARNESS: CAPA TRANSFORMER COMPLETA Q8_0
// ====================================================================

int main(void) {
    printf("====================================================================\n");
    printf("  FULL TRANSFORMER DECODER LAYER Q8_0 (W8A32) HARNESS\n");
    printf("  11 Pipeline Stages on NVIDIA GeForce GT 750M (Kepler GK107 sm_30)\n");
    printf("  Weight Footprint in VRAM: 47.27 MB (Compressed 3.56x from 168 MB)\n");
    printf("====================================================================\n");

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    char dev_name[256];
    CHECK_CUDA(cuDeviceGetName(dev_name, sizeof(dev_name), dev));
    printf("[+] Target Hardware: %s\n", dev_name);

    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    // Cargar CUBINs oficiales
    CUmodule mod_rms, mod_gemv_q8, mod_rope, mod_attn, mod_swiglu;
    CUfunction f_rmsnorm, f_gemv_q8, f_rope, f_attn, f_swiglu, f_residual;

    CHECK_CUDA(cuModuleLoad(&mod_rms, "rms_norm_official.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&f_rmsnorm, mod_rms, "rms_norm_kernel"));

    CHECK_CUDA(cuModuleLoad(&mod_gemv_q8, "gemv_q8_0_official.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&f_gemv_q8, mod_gemv_q8, "gemv_q8_0_kernel"));

    CHECK_CUDA(cuModuleLoad(&mod_rope, "rope_kcache_official.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&f_rope, mod_rope, "rope_kcache_kernel"));

    CHECK_CUDA(cuModuleLoad(&mod_attn, "attention_decode_official.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&f_attn, mod_attn, "attention_decode_kernel"));

    CHECK_CUDA(cuModuleLoad(&mod_swiglu, "swiglu_official.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&f_swiglu, mod_swiglu, "swiglu_kernel"));
    CHECK_CUDA(cuModuleGetFunction(&f_residual, mod_swiglu, "residual_add_kernel"));

    printf("[+] All 5 kernel modules loaded and resolved.\n");

    // Dimensiones canónicas de TinyLlama-1.1B
    const int D = 2048;
    const int H_q = 32;
    const int H_kv = 4;
    const int head_dim = 64;
    const int D_kv = H_kv * head_dim; // 256
    const int D_mlp = 5632;
    const int pos = 64;
    const int context_len = 65;
    const int max_seq_len = 512;
    const float eps = 1e-5f;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int rope_mode = 0; // rotate_half (HuggingFace oficial para TinyLlama)

    printf("[*] Model Spec: D=%d, H_q=%d, H_kv=%d, D_mlp=%d, pos=%d, context=%d (RoPE mode: rotate_half)\n",
           D, H_q, H_kv, D_mlp, pos, context_len);

    // Dimensiones y tamaños de bloques Q8_0
    const int num_blocks_D = D / 32;       // 64 bloques
    const int num_blocks_mlp = D_mlp / 32; // 176 bloques (¡para Wdown!)

    const size_t sz_Wq_q8    = (size_t)D * num_blocks_D * sizeof(block_q8_0);       // 4.5 MB
    const size_t sz_Wk_q8    = (size_t)D_kv * num_blocks_D * sizeof(block_q8_0);    // 0.56 MB
    const size_t sz_Wv_q8    = (size_t)D_kv * num_blocks_D * sizeof(block_q8_0);    // 0.56 MB
    const size_t sz_Wo_q8    = (size_t)D * num_blocks_D * sizeof(block_q8_0);       // 4.5 MB
    const size_t sz_Wgate_q8 = (size_t)D_mlp * num_blocks_D * sizeof(block_q8_0);   // 12.38 MB
    const size_t sz_Wup_q8   = (size_t)D_mlp * num_blocks_D * sizeof(block_q8_0);   // 12.38 MB
    const size_t sz_Wdown_q8 = (size_t)D * num_blocks_mlp * sizeof(block_q8_0);     // 12.38 MB
    const size_t sz_norm     = (size_t)D * sizeof(float);                           // 8 KB

    const size_t total_q8_bytes = sz_Wq_q8 + sz_Wk_q8 + sz_Wv_q8 + sz_Wo_q8 +
                                  sz_Wgate_q8 + sz_Wup_q8 + sz_Wdown_q8;

    printf("[*] Layer Q8_0 Weight Footprint in VRAM: %.2f MB (vs 168.0 MB FP32)\n",
           (double)total_q8_bytes / (1024.0 * 1024.0));

    // Alojar e inicializar tensores en Host
    block_q8_0 *h_Wq    = (block_q8_0 *)malloc(sz_Wq_q8);
    block_q8_0 *h_Wk    = (block_q8_0 *)malloc(sz_Wk_q8);
    block_q8_0 *h_Wv    = (block_q8_0 *)malloc(sz_Wv_q8);
    block_q8_0 *h_Wo    = (block_q8_0 *)malloc(sz_Wo_q8);
    block_q8_0 *h_Wgate = (block_q8_0 *)malloc(sz_Wgate_q8);
    block_q8_0 *h_Wup   = (block_q8_0 *)malloc(sz_Wup_q8);
    block_q8_0 *h_Wdown = (block_q8_0 *)malloc(sz_Wdown_q8);

    float *h_w_attn = (float *)malloc(sz_norm);
    float *h_w_ffn  = (float *)malloc(sz_norm);

    // Inicializar y cuantizar pesos sintéticos reproducibles con escala realista de LLM (1/sqrt(D))
    srand(2026);
    float *temp_row_D = (float *)malloc(D * sizeof(float));
    float *temp_row_mlp = (float *)malloc(D_mlp * sizeof(float));
    float scale_D = 1.0f / sqrtf((float)D);       // ~0.022
    float scale_mlp = 1.0f / sqrtf((float)D_mlp); // ~0.013

    for (int r = 0; r < D; r++) {
        for (int c = 0; c < D; c++) temp_row_D[c] = (((float)((r + c) % 2000) - 1000.0f) * 0.001f) * scale_D;
        quantize_row_q8_0(h_Wq + (size_t)r * num_blocks_D, temp_row_D, D);
        quantize_row_q8_0(h_Wo + (size_t)r * num_blocks_D, temp_row_D, D);
    }
    for (int r = 0; r < D_kv; r++) {
        for (int c = 0; c < D; c++) temp_row_D[c] = (((float)((r + c) % 2000) - 1000.0f) * 0.001f) * scale_D;
        quantize_row_q8_0(h_Wk + (size_t)r * num_blocks_D, temp_row_D, D);
        quantize_row_q8_0(h_Wv + (size_t)r * num_blocks_D, temp_row_D, D);
    }
    for (int r = 0; r < D_mlp; r++) {
        for (int c = 0; c < D; c++) temp_row_D[c] = (((float)((r + c) % 2000) - 1000.0f) * 0.001f) * scale_D;
        quantize_row_q8_0(h_Wgate + (size_t)r * num_blocks_D, temp_row_D, D);
        quantize_row_q8_0(h_Wup   + (size_t)r * num_blocks_D, temp_row_D, D);
    }
    for (int r = 0; r < D; r++) {
        for (int c = 0; c < D_mlp; c++) temp_row_mlp[c] = (((float)((r + c) % 2000) - 1000.0f) * 0.001f) * scale_mlp;
        quantize_row_q8_0(h_Wdown + (size_t)r * num_blocks_mlp, temp_row_mlp, D_mlp);
    }
    free(temp_row_D);
    free(temp_row_mlp);

    for (int i = 0; i < D; i++) {
        h_w_attn[i] = 1.0f + 0.05f * sinf((float)i * 0.01f);
        h_w_ffn[i]  = 1.0f + 0.05f * cosf((float)i * 0.01f);
    }

    // Tabla de frecuencias RoPE (rotate_half)
    float *h_cos = (float *)malloc((head_dim / 2) * sizeof(float));
    float *h_sin = (float *)malloc((head_dim / 2) * sizeof(float));
    for (int i = 0; i < head_dim / 2; i++) {
        double freq = 1.0 / pow(10000.0, (2.0 * i) / (double)head_dim);
        double angle = (double)pos * freq;
        h_cos[i] = (float)cos(angle);
        h_sin[i] = (float)sin(angle);
    }

    // KV Cache y activaciones
    size_t sz_cache = (size_t)H_kv * max_seq_len * head_dim * sizeof(float);
    size_t sz_x = D * sizeof(float);
    size_t sz_Q = H_q * head_dim * sizeof(float);
    size_t sz_KV = H_kv * head_dim * sizeof(float);
    size_t sz_mlp = D_mlp * sizeof(float);

    float *h_x_in = (float *)malloc(sz_x);
    float *h_x_cpu = (float *)malloc(sz_x);
    float *h_x_gpu = (float *)malloc(sz_x);
    float *cpu_K_cache = (float *)calloc(1, sz_cache);
    float *cpu_V_cache = (float *)calloc(1, sz_cache);

    for (int i = 0; i < D; i++) {
        h_x_in[i] = sinf((float)i * 0.02f) * 0.5f;
        h_x_cpu[i] = h_x_in[i];
    }
    for (size_t i = 0; i < (size_t)H_kv * max_seq_len * head_dim; i++) {
        cpu_K_cache[i] = cosf((float)i * 0.01f) * 0.2f;
        cpu_V_cache[i] = sinf((float)i * 0.01f) * 0.2f;
    }

    // ====================================================================
    // ORÁCULO A: REFERENCIA EN CPU (DOUBLE PRECISION EVALUANDO Q8_0)
    // ====================================================================
    printf("[*] Computing CPU Ground Truth (Double Precision with Q8_0 weights)...\n");

    float *cpu_x_norm = (float *)malloc(sz_x);
    float *cpu_Q = (float *)malloc(sz_Q);
    float *cpu_K = (float *)malloc(sz_KV);
    float *cpu_V = (float *)malloc(sz_KV);
    float *cpu_attn_out = (float *)malloc(sz_Q);
    float *cpu_attn_proj = (float *)malloc(sz_x);
    float *cpu_x_ffn_norm = (float *)malloc(sz_x);
    float *cpu_gate = (float *)malloc(sz_mlp);
    float *cpu_up = (float *)malloc(sz_mlp);
    float *cpu_mlp_act = (float *)malloc(sz_mlp);
    float *cpu_down_proj = (float *)malloc(sz_x);

    // 11 Etapas CPU (guardando copias intermedias para diagnóstico exacto)
    float *cpu_x_after_res1 = (float *)malloc(sz_x);
    float *cpu_Q_raw = (float *)malloc(sz_Q);

    cpu_rms_norm(cpu_x_norm, h_x_cpu, h_w_attn, D, eps);
    cpu_gemv_q8_0(cpu_Q, h_Wq, cpu_x_norm, D, D);
    memcpy(cpu_Q_raw, cpu_Q, sz_Q); // Guardar Q antes de RoPE
    cpu_gemv_q8_0(cpu_K, h_Wk, cpu_x_norm, D_kv, D);
    cpu_gemv_q8_0(cpu_V, h_Wv, cpu_x_norm, D_kv, D);
    cpu_rope_kcache(cpu_Q, cpu_K, cpu_V, h_cos, h_sin, cpu_K_cache, cpu_V_cache,
                    pos, H_q, H_kv, head_dim, max_seq_len, rope_mode);
    cpu_attention_decode(cpu_attn_out, cpu_Q, cpu_K_cache, cpu_V_cache,
                         H_q, H_kv, head_dim, max_seq_len, context_len, scale);
    cpu_gemv_q8_0(cpu_attn_proj, h_Wo, cpu_attn_out, D, D);
    cpu_residual_add(h_x_cpu, cpu_attn_proj, D);
    memcpy(cpu_x_after_res1, h_x_cpu, sz_x); // Guardar estado tras Resid 1

    cpu_rms_norm(cpu_x_ffn_norm, h_x_cpu, h_w_ffn, D, eps);
    cpu_gemv_q8_0(cpu_gate, h_Wgate, cpu_x_ffn_norm, D_mlp, D);
    cpu_gemv_q8_0(cpu_up,   h_Wup,   cpu_x_ffn_norm, D_mlp, D);
    cpu_swiglu(cpu_mlp_act, cpu_gate, cpu_up, D_mlp);
    cpu_gemv_q8_0(cpu_down_proj, h_Wdown, cpu_mlp_act, D, D_mlp);
    cpu_residual_add(h_x_cpu, cpu_down_proj, D);

    printf("[+] CPU Layer forward pass finished cleanly.\n");

    // ====================================================================
    // ALOJAR EN GPU VRAM: EMPAQUETADO CONTIGUO DE PESOS (1 SOLO BO DE 47.3 MB)
    // ====================================================================
    printf("[*] Allocating unified 47.3 MB Weight BO in GPU VRAM (Nouveau Safe)...\n");

    CUdeviceptr d_weights_q8;
    CHECK_CUDA(cuMemAlloc(&d_weights_q8, total_q8_bytes));

    size_t off_Wq    = 0;
    size_t off_Wk    = off_Wq + sz_Wq_q8;
    size_t off_Wv    = off_Wk + sz_Wk_q8;
    size_t off_Wo    = off_Wv + sz_Wv_q8;
    size_t off_Wgate = off_Wo + sz_Wo_q8;
    size_t off_Wup   = off_Wgate + sz_Wgate_q8;
    size_t off_Wdown = off_Wup + sz_Wup_q8;

    CUdeviceptr d_Wq    = d_weights_q8 + off_Wq;
    CUdeviceptr d_Wk    = d_weights_q8 + off_Wk;
    CUdeviceptr d_Wv    = d_weights_q8 + off_Wv;
    CUdeviceptr d_Wo    = d_weights_q8 + off_Wo;
    CUdeviceptr d_Wgate = d_weights_q8 + off_Wgate;
    CUdeviceptr d_Wup   = d_weights_q8 + off_Wup;
    CUdeviceptr d_Wdown = d_weights_q8 + off_Wdown;

    CHECK_CUDA(cuMemcpyHtoD(d_Wq,    h_Wq,    sz_Wq_q8));
    CHECK_CUDA(cuMemcpyHtoD(d_Wk,    h_Wk,    sz_Wk_q8));
    CHECK_CUDA(cuMemcpyHtoD(d_Wv,    h_Wv,    sz_Wv_q8));
    CHECK_CUDA(cuMemcpyHtoD(d_Wo,    h_Wo,    sz_Wo_q8));
    CHECK_CUDA(cuMemcpyHtoD(d_Wgate, h_Wgate, sz_Wgate_q8));
    CHECK_CUDA(cuMemcpyHtoD(d_Wup,   h_Wup,   sz_Wup_q8));
    CHECK_CUDA(cuMemcpyHtoD(d_Wdown, h_Wdown, sz_Wdown_q8));

    // Pesos RMSNorm en FP32 (buffer de 16 KB)
    CUdeviceptr d_w_norms;
    CHECK_CUDA(cuMemAlloc(&d_w_norms, 2 * sz_norm));
    CUdeviceptr d_w_attn_dev = d_w_norms;
    CUdeviceptr d_w_ffn_dev  = d_w_norms + sz_norm;
    CHECK_CUDA(cuMemcpyHtoD(d_w_attn_dev, h_w_attn, sz_norm));
    CHECK_CUDA(cuMemcpyHtoD(d_w_ffn_dev,  h_w_ffn,  sz_norm));

    // Frecuencias y KV-Cache
    CUdeviceptr d_cos, d_sin, d_K_cache, d_V_cache;
    CHECK_CUDA(cuMemAlloc(&d_cos, (head_dim / 2) * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_sin, (head_dim / 2) * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_K_cache, sz_cache));
    CHECK_CUDA(cuMemAlloc(&d_V_cache, sz_cache));

    CHECK_CUDA(cuMemcpyHtoD(d_cos, h_cos, (head_dim / 2) * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_sin, h_sin, (head_dim / 2) * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_K_cache, cpu_K_cache, sz_cache));
    CHECK_CUDA(cuMemcpyHtoD(d_V_cache, cpu_V_cache, sz_cache));

    // Activaciones Device
    CUdeviceptr d_x, d_x_norm, d_Q, d_K, d_V, d_attn_out, d_attn_proj;
    CUdeviceptr d_x_ffn_norm, d_gate, d_up, d_mlp_act, d_down_proj;

    CHECK_CUDA(cuMemAlloc(&d_x, sz_x));
    CHECK_CUDA(cuMemAlloc(&d_x_norm, sz_x));
    CHECK_CUDA(cuMemAlloc(&d_Q, sz_Q));
    CHECK_CUDA(cuMemAlloc(&d_K, sz_KV));
    CHECK_CUDA(cuMemAlloc(&d_V, sz_KV));
    CHECK_CUDA(cuMemAlloc(&d_attn_out, sz_Q));
    CHECK_CUDA(cuMemAlloc(&d_attn_proj, sz_x));
    CHECK_CUDA(cuMemAlloc(&d_x_ffn_norm, sz_x));
    CHECK_CUDA(cuMemAlloc(&d_gate, sz_mlp));
    CHECK_CUDA(cuMemAlloc(&d_up, sz_mlp));
    CHECK_CUDA(cuMemAlloc(&d_mlp_act, sz_mlp));
    CHECK_CUDA(cuMemAlloc(&d_down_proj, sz_x));

    CHECK_CUDA(cuMemcpyHtoD(d_x, h_x_in, sz_x));

    // Argumentos de kernels
    int D_arg = D;
    int D_kv_arg = D_kv;
    int D_mlp_arg = D_mlp;
    int num_blocks_D_arg = num_blocks_D;       // 64
    int num_blocks_mlp_arg = num_blocks_mlp;   // 176
    float eps_arg = eps;
    int pos_arg = pos;
    int H_q_arg = H_q;
    int H_kv_arg = H_kv;
    int head_dim_arg = head_dim;
    int max_seq_len_arg = max_seq_len;
    int context_len_arg = context_len;
    float scale_arg = scale;
    int mode_arg = rope_mode; // 0 (rotate_half)

    void *args_norm1[]  = { &d_x_norm, &d_x, &d_w_attn_dev, &D_arg, &eps_arg };
    void *args_Wq[]     = { &d_Q, &d_Wq, &d_x_norm, &D_arg, &num_blocks_D_arg };
    void *args_Wk[]     = { &d_K, &d_Wk, &d_x_norm, &D_kv_arg, &num_blocks_D_arg };
    void *args_Wv[]     = { &d_V, &d_Wv, &d_x_norm, &D_kv_arg, &num_blocks_D_arg };
    void *args_rope[]   = { &d_Q, &d_K, &d_V, &d_cos, &d_sin, &d_K_cache, &d_V_cache,
                           &pos_arg, &H_q_arg, &H_kv_arg, &head_dim_arg, &max_seq_len_arg, &mode_arg };
    void *args_attn[]   = { &d_attn_out, &d_Q, &d_K_cache, &d_V_cache,
                           &H_q_arg, &H_kv_arg, &head_dim_arg, &max_seq_len_arg, &context_len_arg, &scale_arg };
    void *args_Wo[]     = { &d_attn_proj, &d_Wo, &d_attn_out, &D_arg, &num_blocks_D_arg };
    void *args_res1[]   = { &d_x, &d_attn_proj, &D_arg };
    void *args_norm2[]  = { &d_x_ffn_norm, &d_x, &d_w_ffn_dev, &D_arg, &eps_arg };
    void *args_Wgate[]  = { &d_gate, &d_Wgate, &d_x_ffn_norm, &D_mlp_arg, &num_blocks_D_arg };
    void *args_Wup[]    = { &d_up, &d_Wup, &d_x_ffn_norm, &D_mlp_arg, &num_blocks_D_arg };
    void *args_swiglu[] = { &d_mlp_act, &d_gate, &d_up, &D_mlp_arg };
    void *args_Wdown[]  = { &d_down_proj, &d_Wdown, &d_mlp_act, &D_arg, &num_blocks_mlp_arg };
    void *args_res2[]   = { &d_x, &d_down_proj, &D_arg };

    // ====================================================================
    // EJECUCIÓN PIPELINE COMPLETO EN SILICIO KEPLER (11 ETAPAS Q8_0)
    // ====================================================================
    printf("[*] Executing 11-step Q8_0 Layer Pipeline on GeForce GT 750M...\n");

    // 1. RMSNorm 1
    CHECK_CUDA(cuLaunchKernel(f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_norm1, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_x_gpu, d_x_norm, sz_x));
    float diff_norm1 = 0.0f;
    for (int i = 0; i < D; i++) { float d = fabsf(h_x_gpu[i] - cpu_x_norm[i]); if (d > diff_norm1) diff_norm1 = d; }
    printf("    [Step 1  RMSNorm 1] Max Diff: %e\n", diff_norm1);

    // 2. Q, K, V (GEMV Q8_0)
    CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wq, NULL));
    CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D_kv + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wk, NULL));
    CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D_kv + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wv, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_x_gpu, d_Q, sz_Q));
    float diff_q = 0.0f;
    for (int i = 0; i < D; i++) { float d = fabsf(h_x_gpu[i] - cpu_Q_raw[i]); if (d > diff_q) diff_q = d; }
    printf("    [Step 2  GEMV Q   ] Max Diff: %e\n", diff_q);

    // 3. Fused RoPE + KV Cache (rotate_half)
    CHECK_CUDA(cuLaunchKernel(f_rope, H_q, 1, 1, head_dim / 2, 1, 1, 0, 0, args_rope, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_x_gpu, d_Q, sz_Q));
    float diff_rope = 0.0f;
    for (int i = 0; i < D; i++) { float d = fabsf(h_x_gpu[i] - cpu_Q[i]); if (d > diff_rope) diff_rope = d; }
    printf("    [Step 3  RoPE Q   ] Max Diff: %e\n", diff_rope);

    // 4. FlashAttention Online Softmax
    CHECK_CUDA(cuLaunchKernel(f_attn, H_q, 1, 1, 32, 1, 1, 0, 0, args_attn, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_x_gpu, d_attn_out, sz_Q));
    float diff_attn = 0.0f;
    for (int i = 0; i < D; i++) { float d = fabsf(h_x_gpu[i] - cpu_attn_out[i]); if (d > diff_attn) diff_attn = d; }
    printf("    [Step 4  FlashAttn] Max Diff: %e\n", diff_attn);

    // 5. Wo (GEMV Q8_0)
    CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wo, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_x_gpu, d_attn_proj, sz_x));
    float diff_wo = 0.0f;
    for (int i = 0; i < D; i++) { float d = fabsf(h_x_gpu[i] - cpu_attn_proj[i]); if (d > diff_wo) diff_wo = d; }
    printf("    [Step 5  GEMV Wo  ] Max Diff: %e\n", diff_wo);

    // 6. Residual 1
    CHECK_CUDA(cuLaunchKernel(f_residual, (D + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_res1, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_x_gpu, d_x, sz_x));
    float diff_res1 = 0.0f;
    for (int i = 0; i < D; i++) { float d = fabsf(h_x_gpu[i] - cpu_x_after_res1[i]); if (d > diff_res1) diff_res1 = d; }
    printf("    [Step 6  Resid 1  ] Max Diff: %e\n", diff_res1);

    // 7. RMSNorm 2
    CHECK_CUDA(cuLaunchKernel(f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_norm2, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_x_gpu, d_x_ffn_norm, sz_x));
    float diff_norm2 = 0.0f;
    for (int i = 0; i < D; i++) { float d = fabsf(h_x_gpu[i] - cpu_x_ffn_norm[i]); if (d > diff_norm2) diff_norm2 = d; }
    printf("    [Step 7  RMSNorm 2] Max Diff: %e\n", diff_norm2);

    // 8. Wgate, Wup (GEMV Q8_0)
    CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D_mlp + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wgate, NULL));
    CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D_mlp + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wup, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    float *h_mlp_buf = (float *)malloc(sz_mlp);
    CHECK_CUDA(cuMemcpyDtoH(h_mlp_buf, d_gate, sz_mlp));
    float diff_gate = 0.0f;
    for (int i = 0; i < D_mlp; i++) { float d = fabsf(h_mlp_buf[i] - cpu_gate[i]); if (d > diff_gate) diff_gate = d; }
    printf("    [Step 8a GEMV Gate] Max Diff: %e\n", diff_gate);
    CHECK_CUDA(cuMemcpyDtoH(h_mlp_buf, d_up, sz_mlp));
    float diff_up = 0.0f;
    for (int i = 0; i < D_mlp; i++) { float d = fabsf(h_mlp_buf[i] - cpu_up[i]); if (d > diff_up) diff_up = d; }
    printf("    [Step 8b GEMV Up  ] Max Diff: %e\n", diff_up);

    // 9. SwiGLU
    CHECK_CUDA(cuLaunchKernel(f_swiglu, (D_mlp + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_swiglu, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_mlp_buf, d_mlp_act, sz_mlp));
    float diff_swiglu = 0.0f;
    for (int i = 0; i < D_mlp; i++) { float d = fabsf(h_mlp_buf[i] - cpu_mlp_act[i]); if (d > diff_swiglu) diff_swiglu = d; }
    printf("    [Step 9  SwiGLU   ] Max Diff: %e (sample CPU[0]=%f, GPU[0]=%f)\n", diff_swiglu, cpu_mlp_act[0], h_mlp_buf[0]);
    free(h_mlp_buf);

    // 10. Wdown (GEMV Q8_0 con K=5632 / 176 bloques)
    CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wdown, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_x_gpu, d_down_proj, sz_x));
    float diff_down = 0.0f;
    for (int i = 0; i < D; i++) { float d = fabsf(h_x_gpu[i] - cpu_down_proj[i]); if (d > diff_down) diff_down = d; }
    printf("    [Step 10 GEMV Down] Max Diff: %e (sample CPU[0]=%f, GPU[0]=%f)\n", diff_down, cpu_down_proj[0], h_x_gpu[0]);

    // 11. Residual 2
    CHECK_CUDA(cuLaunchKernel(f_residual, (D + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_res2, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_x_gpu, d_x, sz_x));

    // ====================================================================
    // VALIDACIÓN NUMÉRICA: ORÁCULO A (SILICIO GPU VS CPU DOUBLE Q8_0)
    // ====================================================================
    int nan_count = 0;
    int inf_count = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    double dot_gpu_cpu = 0.0, norm_gpu = 0.0, norm_cpu = 0.0;

    for (int i = 0; i < D; i++) {
        if (isnan(h_x_gpu[i])) nan_count++;
        if (isinf(h_x_gpu[i])) inf_count++;
        float diff = fabsf(h_x_gpu[i] - h_x_cpu[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;

        float denom = fabsf(h_x_cpu[i]);
        float rel = (denom > 1e-5f) ? (diff / denom) : diff;
        if (rel > max_rel_diff) max_rel_diff = rel;

        dot_gpu_cpu += (double)h_x_gpu[i] * (double)h_x_cpu[i];
        norm_gpu += (double)h_x_gpu[i] * (double)h_x_gpu[i];
        norm_cpu += (double)h_x_cpu[i] * (double)h_x_cpu[i];
    }
    double cosine_sim = dot_gpu_cpu / (sqrt(norm_gpu) * sqrt(norm_cpu));

    printf("\n[*] End-to-End Accuracy Assessment (Oráculo A: GPU vs CPU Q8_0):\n");
    printf("    Non-finite Values  : %d (Expected: 0)\n", nan_count + inf_count);
    printf("    Max Absolute Error : %e\n", max_abs_diff);
    printf("    Max Relative Error : %e\n", max_rel_diff);
    printf("    Cosine Similarity  : %.8f\n", cosine_sim);

    if (nan_count > 0 || inf_count > 0 || cosine_sim < 0.9999 || max_rel_diff > 5e-4f) {
        fprintf(stderr, "[-] FULL Q8_0 LAYER ACCURACY VERIFICATION FAILED!\n");
        return 1;
    }
    printf("[+] ORACLE A PASSED! GPU matches CPU within FP32 machine precision tolerance.\n");

    // ====================================================================
    // BENCHMARK DE RENDIMIENTO (20 ITERACIONES AMORTIZADAS)
    // ====================================================================
    const int iterations = 20;
    printf("\n[*] Running benchmark (%d full Q8_0 layer iterations)...\n", iterations);

    double t_start = get_time_sec();
    for (int it = 0; it < iterations; it++) {
        CHECK_CUDA(cuLaunchKernel(f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_norm1, NULL));
        CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wq, NULL));
        CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D_kv + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wk, NULL));
        CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D_kv + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wv, NULL));
        CHECK_CUDA(cuLaunchKernel(f_rope, H_q, 1, 1, head_dim / 2, 1, 1, 0, 0, args_rope, NULL));
        CHECK_CUDA(cuLaunchKernel(f_attn, H_q, 1, 1, 32, 1, 1, 0, 0, args_attn, NULL));
        CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wo, NULL));
        CHECK_CUDA(cuLaunchKernel(f_residual, (D + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_res1, NULL));
        CHECK_CUDA(cuLaunchKernel(f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_norm2, NULL));
        CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D_mlp + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wgate, NULL));
        CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D_mlp + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wup, NULL));
        CHECK_CUDA(cuLaunchKernel(f_swiglu, (D_mlp + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_swiglu, NULL));
        CHECK_CUDA(cuLaunchKernel(f_gemv_q8, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wdown, NULL));
        CHECK_CUDA(cuLaunchKernel(f_residual, (D + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_res2, NULL));
    }
    CHECK_CUDA(cuCtxSynchronize());
    double t_end = get_time_sec();

    double total_time_ms = (t_end - t_start) * 1000.0;
    double avg_latency_ms = total_time_ms / (double)iterations;
    double estimated_22_layer_ms = avg_latency_ms * 22.0 + 44.0; // + lm_head

    printf("--------------------------------------------------------------------\n");
    printf("  PERFORMANCE METRICS (Complete Transformer Decoder Layer Q8_0):\n");
    printf("  Total Time (%d passes) : %.3f ms\n", iterations, total_time_ms);
    printf("  Average Latency / Layer: %.2f ms (vs 29.33 ms FP32)\n", avg_latency_ms);
    printf("  Throughput             : %.2f layers/second\n", 1000.0 / avg_latency_ms);
    printf("  Estimated 22-Layer Step: %.1f ms / token (%.2f tokens/sec)\n",
           estimated_22_layer_ms, 1000.0 / estimated_22_layer_ms);
    printf("--------------------------------------------------------------------\n");

    // Limpieza
    cuMemFree(d_weights_q8);
    cuMemFree(d_w_norms);
    cuMemFree(d_cos); cuMemFree(d_sin); cuMemFree(d_K_cache); cuMemFree(d_V_cache);
    cuMemFree(d_x); cuMemFree(d_x_norm); cuMemFree(d_Q); cuMemFree(d_K); cuMemFree(d_V);
    cuMemFree(d_attn_out); cuMemFree(d_attn_proj); cuMemFree(d_x_ffn_norm);
    cuMemFree(d_gate); cuMemFree(d_up); cuMemFree(d_mlp_act); cuMemFree(d_down_proj);

    free(h_Wq); free(h_Wk); free(h_Wv); free(h_Wo);
    free(h_Wgate); free(h_Wup); free(h_Wdown);
    free(h_w_attn); free(h_w_ffn);
    free(cpu_K_cache); free(cpu_V_cache);
    free(h_cos); free(h_sin); free(h_x_in); free(h_x_cpu); free(h_x_gpu);
    free(cpu_x_norm); free(cpu_Q); free(cpu_K); free(cpu_V);
    free(cpu_attn_out); free(cpu_attn_proj); free(cpu_x_ffn_norm);
    free(cpu_gate); free(cpu_up); free(cpu_mlp_act); free(cpu_down_proj);

    cuModuleUnload(mod_rms); cuModuleUnload(mod_gemv_q8); cuModuleUnload(mod_rope);
    cuModuleUnload(mod_attn); cuModuleUnload(mod_swiglu);
    cuCtxDestroy(ctx);

    printf("[+] Full Q8_0 Transformer Layer test finished cleanly with 0 errors.\n");
    return 0;
}
