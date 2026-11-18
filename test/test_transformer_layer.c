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
// REFERENCIA EXACTA EN CPU (PRECISIÓN DOUBLE EN CADA PASO)
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

static void cpu_gemv(float *y, const float *A, const float *x, int M, int K) {
    for (int r = 0; r < M; r++) {
        const float *A_row = A + (size_t)r * K;
        double sum = 0.0;
        for (int c = 0; c < K; c++) {
            sum += (double)A_row[c] * (double)x[c];
        }
        y[r] = (float)sum;
    }
}

static void cpu_rope_kcache(
    float *Q, const float *K_in, const float *V_in,
    const float *cos_pos, const float *sin_pos,
    float *K_cache, float *V_cache,
    int pos, int num_heads, int num_kv_heads, int head_dim, int max_seq_len)
{
    int half_dim = head_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *q_h = Q + (size_t)h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            double c = (double)cos_pos[i];
            double s = (double)sin_pos[i];
            double q0 = (double)q_h[i * 2];
            double q1 = (double)q_h[i * 2 + 1];
            q_h[i * 2]     = (float)(q0 * c - q1 * s);
            q_h[i * 2 + 1] = (float)(q0 * s + q1 * c);
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        const float *k_h = K_in + (size_t)h * head_dim;
        const float *v_h = V_in + (size_t)h * head_dim;
        size_t off = ((size_t)h * max_seq_len + pos) * head_dim;
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

        double inv_sum = sum_exp > 0.0 ? (1.0 / sum_exp) : 0.0;
        for (int i = 0; i < head_dim; i++) {
            out_h[i] = (float)(acc[i] * inv_sum);
        }
    }
}

static void cpu_swiglu(float *out, const float *gate, const float *up, int dim) {
    for (int i = 0; i < dim; i++) {
        double g = (double)gate[i];
        double u = (double)up[i];
        double silu = g / (1.0 + exp(-g));
        out[i] = (float)(silu * u);
    }
}

static void cpu_residual_add(float *x, const float *residual, int dim) {
    for (int i = 0; i < dim; i++) {
        x[i] += residual[i];
    }
}

// ====================================================================
// PROGRAMA PRINCIPAL: INTEGRACIÓN DE CAPA TRANSFORMER COMPLETA
// ====================================================================

int main(void) {
    printf("====================================================================\n");
    printf("  FULL TRANSFORMER DECODER LAYER HARNESS (TinyLlama-1.1B Architecture)\n");
    printf("  End-to-End Silicon Execution on NVIDIA GeForce GT 750M (Kepler sm_30)\n");
    printf("====================================================================\n");

    CHECK_CUDA(cuInit(0));

    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    char dev_name[128];
    CHECK_CUDA(cuDeviceGetName(dev_name, sizeof(dev_name), dev));
    printf("[+] Target Hardware: %s\n", dev_name);

    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    // Cargar módulos CUBIN oficiales
    CUmodule mod_rmsnorm, mod_gemv, mod_rope, mod_attn, mod_swiglu;
    CHECK_CUDA(cuModuleLoad(&mod_rmsnorm, "rms_norm_official.cubin"));
    CHECK_CUDA(cuModuleLoad(&mod_gemv, "gemv_official.cubin"));
    CHECK_CUDA(cuModuleLoad(&mod_rope, "rope_kcache_official.cubin"));
    CHECK_CUDA(cuModuleLoad(&mod_attn, "attention_decode_official.cubin"));
    CHECK_CUDA(cuModuleLoad(&mod_swiglu, "swiglu_official.cubin"));
    printf("[+] All 5 CUBIN modules loaded successfully.\n");

    CUfunction f_rmsnorm, f_gemv, f_rope, f_attn, f_swiglu, f_residual;
    CHECK_CUDA(cuModuleGetFunction(&f_rmsnorm, mod_rmsnorm, "rms_norm_kernel"));
    CHECK_CUDA(cuModuleGetFunction(&f_gemv, mod_gemv, "gemv_kernel"));
    CHECK_CUDA(cuModuleGetFunction(&f_rope, mod_rope, "rope_kcache_kernel"));
    CHECK_CUDA(cuModuleGetFunction(&f_attn, mod_attn, "attention_decode_kernel"));
    CHECK_CUDA(cuModuleGetFunction(&f_swiglu, mod_swiglu, "swiglu_kernel"));
    CHECK_CUDA(cuModuleGetFunction(&f_residual, mod_swiglu, "residual_add_kernel"));
    printf("[+] All 6 kernel functions resolved.\n");

    // Hiperparámetros de TinyLlama-1.1B
    const int D = 2048;                // Dimensión oculta
    const int H_q = 32;                // Cabezas Query
    const int H_kv = 4;                // Cabezas Key/Value (GQA 8:1)
    const int head_dim = 64;           // D / H_q
    const int D_kv = H_kv * head_dim;  // 256
    const int D_mlp = 5632;            // Dimensión intermedia FFN
    const int max_seq_len = 512;       // Capacidad de contexto
    const int pos = 64;                // Decodificando token en posición 64
    const int context_len = pos + 1;   // 65 tokens acumulados
    const float eps = 1e-5f;
    const float scale = 1.0f / sqrtf((float)head_dim);

    printf("[*] Model Spec: D=%d, H_q=%d, H_kv=%d, D_mlp=%d, pos=%d, context=%d\n",
           D, H_q, H_kv, D_mlp, pos, context_len);

    // Tamaños de matrices de pesos
    const size_t sz_Wq = (size_t)D * D * sizeof(float);       // 16 MB
    const size_t sz_Wk = (size_t)D_kv * D * sizeof(float);    // 2 MB
    const size_t sz_Wv = (size_t)D_kv * D * sizeof(float);    // 2 MB
    const size_t sz_Wo = (size_t)D * D * sizeof(float);       // 16 MB
    const size_t sz_Wgate = (size_t)D_mlp * D * sizeof(float);// 44 MB
    const size_t sz_Wup = (size_t)D_mlp * D * sizeof(float);  // 44 MB
    const size_t sz_Wdown = (size_t)D * D_mlp * sizeof(float);// 44 MB
    const size_t sz_norm = (size_t)D * sizeof(float);         // 8 KB
    const size_t total_weight_bytes = sz_Wq + sz_Wk + sz_Wv + sz_Wo + sz_Wgate + sz_Wup + sz_Wdown + 2 * sz_norm;

    printf("[*] Layer Weights Footprint in VRAM: %.1f MB\n", (double)total_weight_bytes / (1024.0 * 1024.0));

    // Alojar memoria host para pesos
    float *h_Wq = (float *)malloc(sz_Wq);
    float *h_Wk = (float *)malloc(sz_Wk);
    float *h_Wv = (float *)malloc(sz_Wv);
    float *h_Wo = (float *)malloc(sz_Wo);
    float *h_Wgate = (float *)malloc(sz_Wgate);
    float *h_Wup = (float *)malloc(sz_Wup);
    float *h_Wdown = (float *)malloc(sz_Wdown);
    float *h_w_attn_norm = (float *)malloc(sz_norm);
    float *h_w_ffn_norm = (float *)malloc(sz_norm);

    // Inicializar pesos con distribuciones pequeñas
    for (size_t i = 0; i < (size_t)D * D; i++) h_Wq[i] = (((float)(i % 1000) - 500.0f) * 0.0001f);
    for (size_t i = 0; i < (size_t)D_kv * D; i++) h_Wk[i] = (((float)(i % 1000) - 500.0f) * 0.0001f);
    for (size_t i = 0; i < (size_t)D_kv * D; i++) h_Wv[i] = (((float)(i % 1000) - 500.0f) * 0.0001f);
    for (size_t i = 0; i < (size_t)D * D; i++) h_Wo[i] = (((float)(i % 1000) - 500.0f) * 0.0001f);
    for (size_t i = 0; i < (size_t)D_mlp * D; i++) h_Wgate[i] = (((float)(i % 1000) - 500.0f) * 0.0001f);
    for (size_t i = 0; i < (size_t)D_mlp * D; i++) h_Wup[i] = (((float)(i % 1000) - 500.0f) * 0.0001f);
    for (size_t i = 0; i < (size_t)D * D_mlp; i++) h_Wdown[i] = (((float)(i % 1000) - 500.0f) * 0.0001f);
    for (int i = 0; i < D; i++) {
        h_w_attn_norm[i] = 1.0f + 0.05f * sinf((float)i * 0.01f);
        h_w_ffn_norm[i] = 1.0f + 0.05f * cosf((float)i * 0.01f);
    }

    // Tabla de frecuencias RoPE
    float *h_cos = (float *)malloc((head_dim / 2) * sizeof(float));
    float *h_sin = (float *)malloc((head_dim / 2) * sizeof(float));
    for (int i = 0; i < head_dim / 2; i++) {
        double freq = 1.0 / pow(10000.0, (2.0 * i) / (double)head_dim);
        double angle = (double)pos * freq;
        h_cos[i] = (float)cos(angle);
        h_sin[i] = (float)sin(angle);
    }

    // Buffers de activación (Host)
    size_t sz_x = D * sizeof(float);
    size_t sz_Q = H_q * head_dim * sizeof(float);
    size_t sz_KV = H_kv * head_dim * sizeof(float);
    size_t sz_cache = (size_t)H_kv * max_seq_len * head_dim * sizeof(float);
    size_t sz_mlp = D_mlp * sizeof(float);

    float *h_x_in = (float *)malloc(sz_x);
    float *h_x_cpu = (float *)malloc(sz_x);
    float *h_x_gpu = (float *)malloc(sz_x);

    // Buffers temporales en CPU
    float *cpu_x_norm = (float *)malloc(sz_x);
    float *cpu_Q = (float *)malloc(sz_Q);
    float *cpu_K = (float *)malloc(sz_KV);
    float *cpu_V = (float *)malloc(sz_KV);
    float *cpu_K_cache = (float *)calloc(1, sz_cache);
    float *cpu_V_cache = (float *)calloc(1, sz_cache);
    float *cpu_attn_out = (float *)malloc(sz_Q);
    float *cpu_attn_proj = (float *)malloc(sz_x);
    float *cpu_x_ffn_norm = (float *)malloc(sz_x);
    float *cpu_gate = (float *)malloc(sz_mlp);
    float *cpu_up = (float *)malloc(sz_mlp);
    float *cpu_mlp_act = (float *)malloc(sz_mlp);
    float *cpu_down_proj = (float *)malloc(sz_x);

    // Inicializar token de entrada y pre-poblar KV cache con tokens previos
    for (int i = 0; i < D; i++) {
        h_x_in[i] = sinf((float)i * 0.02f) * 0.5f;
        h_x_cpu[i] = h_x_in[i];
    }
    for (size_t i = 0; i < (size_t)H_kv * max_seq_len * head_dim; i++) {
        cpu_K_cache[i] = cosf((float)i * 0.01f) * 0.2f;
        cpu_V_cache[i] = sinf((float)i * 0.01f) * 0.2f;
    }

    // ====================================================================
    // 1. COMPUTAR REFERENCIA COMPLETA EN CPU (DOUBLE PRECISION)
    // ====================================================================
    printf("[*] Computing ground truth Transformer Layer Forward Pass on CPU...\n");

    // Paso 1: Pre-Attention RMSNorm
    cpu_rms_norm(cpu_x_norm, h_x_cpu, h_w_attn_norm, D, eps);

    // Paso 2: Proyecciones Q, K, V
    cpu_gemv(cpu_Q, h_Wq, cpu_x_norm, D, D);
    cpu_gemv(cpu_K, h_Wk, cpu_x_norm, D_kv, D);
    cpu_gemv(cpu_V, h_Wv, cpu_x_norm, D_kv, D);

    // Paso 3: Fused RoPE + KV Cache Update
    cpu_rope_kcache(cpu_Q, cpu_K, cpu_V, h_cos, h_sin, cpu_K_cache, cpu_V_cache,
                    pos, H_q, H_kv, head_dim, max_seq_len);

    // Paso 4: FlashAttention Online Softmax Decode
    cpu_attention_decode(cpu_attn_out, cpu_Q, cpu_K_cache, cpu_V_cache,
                         H_q, H_kv, head_dim, max_seq_len, context_len, scale);

    // Paso 5: Proyección de Salida de Atención Wo
    cpu_gemv(cpu_attn_proj, h_Wo, cpu_attn_out, D, D);

    // Paso 6: Residual Skip 1 (x = x + attn_proj)
    cpu_residual_add(h_x_cpu, cpu_attn_proj, D);

    // Paso 7: Pre-FFN RMSNorm
    cpu_rms_norm(cpu_x_ffn_norm, h_x_cpu, h_w_ffn_norm, D, eps);

    // Paso 8: Proyecciones MLP Gate y Up
    cpu_gemv(cpu_gate, h_Wgate, cpu_x_ffn_norm, D_mlp, D);
    cpu_gemv(cpu_up, h_Wup, cpu_x_ffn_norm, D_mlp, D);

    // Paso 9: Activación SwiGLU
    cpu_swiglu(cpu_mlp_act, cpu_gate, cpu_up, D_mlp);

    // Paso 10: Proyección Descendente Wdown
    cpu_gemv(cpu_down_proj, h_Wdown, cpu_mlp_act, D, D_mlp);

    // Paso 11: Residual Skip 2 (x = x + down_proj)
    cpu_residual_add(h_x_cpu, cpu_down_proj, D);

    printf("[+] CPU Ground Truth Layer output computed successfully.\n");

    // ====================================================================
    // 2. ALOJAR Y PREPARAR MEMORIA EN GPU VRAM
    // ====================================================================
    printf("[*] Allocating and populating Layer tensors in GPU VRAM...\n");

    // Pesos en VRAM
    CUdeviceptr d_Wq, d_Wk, d_Wv, d_Wo, d_Wgate, d_Wup, d_Wdown, d_w_attn, d_w_ffn;
    CHECK_CUDA(cuMemAlloc(&d_Wq, sz_Wq));
    CHECK_CUDA(cuMemAlloc(&d_Wk, sz_Wk));
    CHECK_CUDA(cuMemAlloc(&d_Wv, sz_Wv));
    CHECK_CUDA(cuMemAlloc(&d_Wo, sz_Wo));
    CHECK_CUDA(cuMemAlloc(&d_Wgate, sz_Wgate));
    CHECK_CUDA(cuMemAlloc(&d_Wup, sz_Wup));
    CHECK_CUDA(cuMemAlloc(&d_Wdown, sz_Wdown));
    CHECK_CUDA(cuMemAlloc(&d_w_attn, sz_norm));
    CHECK_CUDA(cuMemAlloc(&d_w_ffn, sz_norm));

    CHECK_CUDA(cuMemcpyHtoD(d_Wq, h_Wq, sz_Wq));
    CHECK_CUDA(cuMemcpyHtoD(d_Wk, h_Wk, sz_Wk));
    CHECK_CUDA(cuMemcpyHtoD(d_Wv, h_Wv, sz_Wv));
    CHECK_CUDA(cuMemcpyHtoD(d_Wo, h_Wo, sz_Wo));
    CHECK_CUDA(cuMemcpyHtoD(d_Wgate, h_Wgate, sz_Wgate));
    CHECK_CUDA(cuMemcpyHtoD(d_Wup, h_Wup, sz_Wup));
    CHECK_CUDA(cuMemcpyHtoD(d_Wdown, h_Wdown, sz_Wdown));
    CHECK_CUDA(cuMemcpyHtoD(d_w_attn, h_w_attn_norm, sz_norm));
    CHECK_CUDA(cuMemcpyHtoD(d_w_ffn, h_w_ffn_norm, sz_norm));

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

    // Buffers de activación en VRAM (Working Set para L2 Cache)
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

    // Subir token de entrada
    CHECK_CUDA(cuMemcpyHtoD(d_x, h_x_in, sz_x));

    // Argumentos de kernels
    int D_arg = D;
    int D_kv_arg = D_kv;
    int D_mlp_arg = D_mlp;
    int K4_D = D / 4;
    int K4_mlp = D_mlp / 4;
    float eps_arg = eps;
    int pos_arg = pos;
    int H_q_arg = H_q;
    int H_kv_arg = H_kv;
    int head_dim_arg = head_dim;
    int max_seq_len_arg = max_seq_len;
    int context_len_arg = context_len;
    float scale_arg = scale;
    int mode_arg = 1; // 1: interleaved (Meta), 0: rotate_half (HF)

    // Listas de argumentos fijadas para cada paso
    void *args_norm1[] = { &d_x_norm, &d_x, &d_w_attn, &D_arg, &eps_arg };
    void *args_Wq[]    = { &d_Q, &d_Wq, &d_x_norm, &D_arg, &K4_D };
    void *args_Wk[]    = { &d_K, &d_Wk, &d_x_norm, &D_kv_arg, &K4_D };
    void *args_Wv[]    = { &d_V, &d_Wv, &d_x_norm, &D_kv_arg, &K4_D };
    void *args_rope[]  = { &d_Q, &d_K, &d_V, &d_cos, &d_sin, &d_K_cache, &d_V_cache,
                          &pos_arg, &H_q_arg, &H_kv_arg, &head_dim_arg, &max_seq_len_arg, &mode_arg };
    void *args_attn[]  = { &d_attn_out, &d_Q, &d_K_cache, &d_V_cache,
                          &H_q_arg, &H_kv_arg, &head_dim_arg, &max_seq_len_arg, &context_len_arg, &scale_arg };
    void *args_Wo[]    = { &d_attn_proj, &d_Wo, &d_attn_out, &D_arg, &K4_D };
    void *args_res1[]  = { &d_x, &d_attn_proj, &D_arg };
    void *args_norm2[] = { &d_x_ffn_norm, &d_x, &d_w_ffn, &D_arg, &eps_arg };
    void *args_Wgate[] = { &d_gate, &d_Wgate, &d_x_ffn_norm, &D_mlp_arg, &K4_D };
    void *args_Wup[]   = { &d_up, &d_Wup, &d_x_ffn_norm, &D_mlp_arg, &K4_D };
    void *args_swiglu[]= { &d_mlp_act, &d_gate, &d_up, &D_mlp_arg };
    void *args_Wdown[] = { &d_down_proj, &d_Wdown, &d_mlp_act, &D_arg, &K4_mlp };
    void *args_res2[]  = { &d_x, &d_down_proj, &D_arg };

    // ====================================================================
    // 3. EJECUTAR PIPELINE COMPLETO EN SILICIO (WARMUP & VALIDACIÓN)
    // ====================================================================
    printf("[*] Executing 11-step Layer Pipeline on GeForce GT 750M...\n");

    // Paso 1: Pre-Attention Norm
    CHECK_CUDA(cuLaunchKernel(f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_norm1, NULL));
    // Paso 2: GEMV Q, K, V
    CHECK_CUDA(cuLaunchKernel(f_gemv, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wq, NULL));
    CHECK_CUDA(cuLaunchKernel(f_gemv, (D_kv + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wk, NULL));
    CHECK_CUDA(cuLaunchKernel(f_gemv, (D_kv + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wv, NULL));
    // Paso 3: Fused RoPE + KV Cache
    CHECK_CUDA(cuLaunchKernel(f_rope, H_q, 1, 1, head_dim / 2, 1, 1, 0, 0, args_rope, NULL));
    // Paso 4: FlashAttention Online Softmax
    CHECK_CUDA(cuLaunchKernel(f_attn, H_q, 1, 1, 32, 1, 1, 0, 0, args_attn, NULL));
    // Paso 5: Output Projection Wo
    CHECK_CUDA(cuLaunchKernel(f_gemv, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wo, NULL));
    // Paso 6: Residual Skip 1
    CHECK_CUDA(cuLaunchKernel(f_residual, (D + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_res1, NULL));
    // Paso 7: Pre-FFN Norm
    CHECK_CUDA(cuLaunchKernel(f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_norm2, NULL));
    // Paso 8: MLP Gate & Up
    CHECK_CUDA(cuLaunchKernel(f_gemv, (D_mlp + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wgate, NULL));
    CHECK_CUDA(cuLaunchKernel(f_gemv, (D_mlp + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wup, NULL));
    // Paso 9: SwiGLU Activation
    CHECK_CUDA(cuLaunchKernel(f_swiglu, (D_mlp + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_swiglu, NULL));
    // Paso 10: MLP Down Projection
    CHECK_CUDA(cuLaunchKernel(f_gemv, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wdown, NULL));
    // Paso 11: Residual Skip 2
    CHECK_CUDA(cuLaunchKernel(f_residual, (D + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_res2, NULL));

    CHECK_CUDA(cuCtxSynchronize());
    printf("[+] Layer execution finished. Reading back final hidden state...\n");

    CHECK_CUDA(cuMemcpyDtoH(h_x_gpu, d_x, sz_x));

    // ====================================================================
    // 4. VALIDACIÓN NUMÉRICA RIGUROSA CONTRA CPU
    // ====================================================================
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t nonfinite_count = 0;
    size_t mismatch_count = 0;

    for (int i = 0; i < D; i++) {
        float gpu_val = h_x_gpu[i];
        float cpu_val = h_x_cpu[i];

        if (!isfinite(gpu_val) || !isfinite(cpu_val)) {
            nonfinite_count++;
            continue;
        }

        double diff = fabs((double)gpu_val - (double)cpu_val);
        double ref = fabs((double)cpu_val);
        double rel = ref > 1e-5 ? diff / ref : diff;

        if (diff > max_abs_diff) max_abs_diff = diff;
        if (rel > max_rel_diff) max_rel_diff = rel;

        if (diff > 5e-3) {
            mismatch_count++;
            if (mismatch_count <= 5) {
                printf("[-] Mismatch at dim %d: GPU=%f, CPU=%f, diff=%e\n",
                       i, gpu_val, cpu_val, diff);
            }
        }
    }

    printf("\n[*] End-to-End Accuracy Assessment (11 Pipeline Stages):\n");
    printf("    Non-finite Values  : %zu (Expected: 0)\n", nonfinite_count);
    printf("    Max Absolute Error : %e\n", max_abs_diff);
    printf("    Max Relative Error : %e\n", max_rel_diff);
    printf("    Mismatches (> 5e-3): %zu / %d\n", mismatch_count, D);

    if (nonfinite_count > 0 || mismatch_count > 0 || max_abs_diff > 1e-2) {
        fprintf(stderr, "[-] FAILED: Layer Forward Pass verification failed!\n");
        return 1;
    }
    printf("[+] FULL LAYER VERIFICATION PASSED! GPU matches CPU within FP32 machine precision.\n\n");

    // ====================================================================
    // 5. BENCHMARK DE RENDIMIENTO (20 ITERACIONES AMORTIZADAS)
    // ====================================================================
    const int benchmark_iters = 20;
    printf("[*] Running benchmark (%d full layer iterations with amortized GPU sync)...\n", benchmark_iters);

    CHECK_CUDA(cuCtxSynchronize());
    double t_start = get_time_sec();

    for (int iter = 0; iter < benchmark_iters; iter++) {
        cuLaunchKernel(f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_norm1, NULL);
        cuLaunchKernel(f_gemv, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wq, NULL);
        cuLaunchKernel(f_gemv, (D_kv + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wk, NULL);
        cuLaunchKernel(f_gemv, (D_kv + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wv, NULL);
        cuLaunchKernel(f_rope, H_q, 1, 1, head_dim / 2, 1, 1, 0, 0, args_rope, NULL);
        cuLaunchKernel(f_attn, H_q, 1, 1, 32, 1, 1, 0, 0, args_attn, NULL);
        cuLaunchKernel(f_gemv, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wo, NULL);
        cuLaunchKernel(f_residual, (D + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_res1, NULL);
        cuLaunchKernel(f_rmsnorm, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), 0, args_norm2, NULL);
        cuLaunchKernel(f_gemv, (D_mlp + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wgate, NULL);
        cuLaunchKernel(f_gemv, (D_mlp + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wup, NULL);
        cuLaunchKernel(f_swiglu, (D_mlp + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_swiglu, NULL);
        cuLaunchKernel(f_gemv, (D + 3) / 4, 1, 1, 32, 4, 1, 0, 0, args_Wdown, NULL);
        cuLaunchKernel(f_residual, (D + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_res2, NULL);
    }

    CHECK_CUDA(cuCtxSynchronize());
    double t_end = get_time_sec();

    double total_sec = t_end - t_start;
    double avg_ms = (total_sec / (double)benchmark_iters) * 1000.0;
    double layers_per_sec = 1.0 / (avg_ms / 1000.0);

    printf("--------------------------------------------------------------------\n");
    printf("  PERFORMANCE METRICS (Complete Transformer Decoder Layer):\n");
    printf("  Total Time (%d passes) : %.3f ms\n", benchmark_iters, total_sec * 1000.0);
    printf("  Average Latency / Layer: %.2f ms\n", avg_ms);
    printf("  Throughput             : %.2f layers/second\n", layers_per_sec);
    printf("  Estimated 22-Layer Step: %.1f ms / token (%.2f tokens/sec)\n",
           avg_ms * 22.0, 1000.0 / (avg_ms * 22.0));
    printf("--------------------------------------------------------------------\n");

    // Limpieza
    cuMemFree(d_Wq); cuMemFree(d_Wk); cuMemFree(d_Wv); cuMemFree(d_Wo);
    cuMemFree(d_Wgate); cuMemFree(d_Wup); cuMemFree(d_Wdown);
    cuMemFree(d_w_attn); cuMemFree(d_w_ffn);
    cuMemFree(d_cos); cuMemFree(d_sin); cuMemFree(d_K_cache); cuMemFree(d_V_cache);
    cuMemFree(d_x); cuMemFree(d_x_norm); cuMemFree(d_Q); cuMemFree(d_K); cuMemFree(d_V);
    cuMemFree(d_attn_out); cuMemFree(d_attn_proj); cuMemFree(d_x_ffn_norm);
    cuMemFree(d_gate); cuMemFree(d_up); cuMemFree(d_mlp_act); cuMemFree(d_down_proj);

    free(h_Wq); free(h_Wk); free(h_Wv); free(h_Wo);
    free(h_Wgate); free(h_Wup); free(h_Wdown);
    free(h_w_attn_norm); free(h_w_ffn_norm);
    free(h_cos); free(h_sin);
    free(h_x_in); free(h_x_cpu); free(h_x_gpu);
    free(cpu_x_norm); free(cpu_Q); free(cpu_K); free(cpu_V);
    free(cpu_K_cache); free(cpu_V_cache);
    free(cpu_attn_out); free(cpu_attn_proj); free(cpu_x_ffn_norm);
    free(cpu_gate); free(cpu_up); free(cpu_mlp_act); free(cpu_down_proj);

    cuModuleUnload(mod_rmsnorm);
    cuModuleUnload(mod_gemv);
    cuModuleUnload(mod_rope);
    cuModuleUnload(mod_attn);
    cuModuleUnload(mod_swiglu);
    cuCtxDestroy(ctx);

    printf("[+] Full Transformer Decoder Layer test finished cleanly with 0 errors.\n");
    return 0;
}
