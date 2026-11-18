#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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

// Referencia en CPU de RoPE + KV-Cache en doble precision
static void cpu_rope_kcache(
    float *Q, const float *K_in, const float *V_in,
    const float *cos_pos, const float *sin_pos,
    float *K_cache, float *V_cache,
    int pos, int num_heads, int num_kv_heads, int head_dim, int max_seq_len)
{
    int half_dim = head_dim / 2;

    // 1. RoPE en Q
    for (int h = 0; h < num_heads; h++) {
        float *q_head = Q + (size_t)h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            double c = (double)cos_pos[i];
            double s = (double)sin_pos[i];
            double q0 = (double)q_head[i * 2];
            double q1 = (double)q_head[i * 2 + 1];
            q_head[i * 2]     = (float)(q0 * c - q1 * s);
            q_head[i * 2 + 1] = (float)(q0 * s + q1 * c);
        }
    }

    // 2. RoPE en K y actualizacion de KV Cache
    for (int h = 0; h < num_kv_heads; h++) {
        const float *k_head = K_in + (size_t)h * head_dim;
        const float *v_head = V_in + (size_t)h * head_dim;
        size_t cache_offset = ((size_t)h * max_seq_len + pos) * head_dim;

        for (int i = 0; i < half_dim; i++) {
            double c = (double)cos_pos[i];
            double s = (double)sin_pos[i];
            double k0 = (double)k_head[i * 2];
            double k1 = (double)k_head[i * 2 + 1];

            K_cache[cache_offset + i * 2]     = (float)(k0 * c - k1 * s);
            K_cache[cache_offset + i * 2 + 1] = (float)(k0 * s + k1 * c);

            V_cache[cache_offset + i * 2]     = v_head[i * 2];
            V_cache[cache_offset + i * 2 + 1] = v_head[i * 2 + 1];
        }
    }
}

int main(void) {
    printf("====================================================================\n");
    printf("  TRANSFORMER INFERENCE KERNEL BENCHMARK: FUSED RoPE + KV-CACHE\n");
    printf("  Interleaved Rotary Embedding & Direct GQA KV-Cache Placement\n");
    printf("  Silicon: NVIDIA GeForce GT 750M (Kepler GK107, sm_30)\n");
    printf("====================================================================\n");

    CHECK_CUDA(cuInit(0));

    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));

    char name[128];
    CHECK_CUDA(cuDeviceGetName(name, sizeof(name), dev));
    printf("[+] Target Device: %s\n", name);

    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK_CUDA(cuModuleLoad(&mod, "rope_kcache_official.cubin"));
    printf("[+] Loaded CUBIN 'rope_kcache_official.cubin' successfully.\n");

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "rope_kcache_kernel"));
    printf("[+] Function 'rope_kcache_kernel' resolved.\n");

    // Parametros reales de TinyLlama-1.1B GQA
    const int num_heads = 32;       // Cabezas Query
    const int num_kv_heads = 4;     // Cabezas Key/Value (GQA)
    const int head_dim = 64;        // Dimension por cabeza (2048 / 32)
    const int half_dim = head_dim / 2; // 32 frecuencias por cabeza
    const int max_seq_len = 512;    // Capacidad de contexto del KV-Cache
    const int pos = 127;            // Token en posicion 127

    const size_t q_elements = (size_t)num_heads * head_dim;      // 2048 floats
    const size_t kv_in_elements = (size_t)num_kv_heads * head_dim; // 256 floats
    const size_t cache_elements = (size_t)num_kv_heads * max_seq_len * head_dim; // 131,072 floats

    const size_t q_bytes = q_elements * sizeof(float);
    const size_t kv_in_bytes = kv_in_elements * sizeof(float);
    const size_t sincos_bytes = half_dim * sizeof(float);
    const size_t cache_bytes = cache_elements * sizeof(float);

    printf("[*] Model Architecture: Q-Heads=%d, KV-Heads=%d (GQA 8:1), HeadDim=%d\n",
           num_heads, num_kv_heads, head_dim);
    printf("[*] KV-Cache Capacity: %d tokens (%zu KB per tensor)\n",
           max_seq_len, cache_bytes / 1024);

    // Buffers host
    float *h_Q_gpu = (float *)malloc(q_bytes);
    float *h_Q_cpu = (float *)malloc(q_bytes);
    float *h_K_in = (float *)malloc(kv_in_bytes);
    float *h_V_in = (float *)malloc(kv_in_bytes);
    float *h_cos = (float *)malloc(sincos_bytes);
    float *h_sin = (float *)malloc(sincos_bytes);
    float *h_K_cache_gpu = (float *)malloc(cache_bytes);
    float *h_K_cache_cpu = (float *)malloc(cache_bytes);
    float *h_V_cache_gpu = (float *)malloc(cache_bytes);
    float *h_V_cache_cpu = (float *)malloc(cache_bytes);

    // Inicializar tensores
    for (size_t i = 0; i < q_elements; i++) {
        h_Q_gpu[i] = sinf((float)i * 0.03f);
        h_Q_cpu[i] = h_Q_gpu[i];
    }
    for (size_t i = 0; i < kv_in_elements; i++) {
        h_K_in[i] = cosf((float)i * 0.05f);
        h_V_in[i] = sinf((float)i * 0.07f);
    }
    // Frecuencias theta_i = 10000^(-2(i-1)/dim)
    for (int i = 0; i < half_dim; i++) {
        double freq = 1.0 / pow(10000.0, (2.0 * i) / (double)head_dim);
        double angle = (double)pos * freq;
        h_cos[i] = (float)cos(angle);
        h_sin[i] = (float)sin(angle);
    }
    memset(h_K_cache_gpu, 0, cache_bytes);
    memset(h_K_cache_cpu, 0, cache_bytes);
    memset(h_V_cache_gpu, 0, cache_bytes);
    memset(h_V_cache_cpu, 0, cache_bytes);

    // Referencia CPU
    printf("[*] Computing ground truth reference on CPU...\n");
    cpu_rope_kcache(h_Q_cpu, h_K_in, h_V_in, h_cos, h_sin,
                    h_K_cache_cpu, h_V_cache_cpu,
                    pos, num_heads, num_kv_heads, head_dim, max_seq_len);

    // Buffers VRAM
    CUdeviceptr d_Q, d_K_in, d_V_in, d_cos, d_sin, d_K_cache, d_V_cache;
    CHECK_CUDA(cuMemAlloc(&d_Q, q_bytes));
    CHECK_CUDA(cuMemAlloc(&d_K_in, kv_in_bytes));
    CHECK_CUDA(cuMemAlloc(&d_V_in, kv_in_bytes));
    CHECK_CUDA(cuMemAlloc(&d_cos, sincos_bytes));
    CHECK_CUDA(cuMemAlloc(&d_sin, sincos_bytes));
    CHECK_CUDA(cuMemAlloc(&d_K_cache, cache_bytes));
    CHECK_CUDA(cuMemAlloc(&d_V_cache, cache_bytes));

    CHECK_CUDA(cuMemcpyHtoD(d_Q, h_Q_gpu, q_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_K_in, h_K_in, kv_in_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_V_in, h_V_in, kv_in_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_cos, h_cos, sincos_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_sin, h_sin, sincos_bytes));
    CHECK_CUDA(cuMemsetD8(d_K_cache, 0, cache_bytes));
    CHECK_CUDA(cuMemsetD8(d_V_cache, 0, cache_bytes));

    // Geometria de lanzamiento: 32 bloques (1 por cada cabeza Q), 32 hilos por bloque
    const unsigned int grid_x = num_heads;
    const unsigned int block_x = half_dim; // 32 hilos por bloque
    int pos_arg = pos;
    int num_heads_arg = num_heads;
    int num_kv_heads_arg = num_kv_heads;
    int head_dim_arg = head_dim;
    int max_seq_len_arg = max_seq_len;
    int mode_arg = 1; // 1: interleaved (Meta), 0: rotate_half (HF)

    void *args[] = {
        &d_Q,
        &d_K_in,
        &d_V_in,
        &d_cos,
        &d_sin,
        &d_K_cache,
        &d_V_cache,
        &pos_arg,
        &num_heads_arg,
        &num_kv_heads_arg,
        &head_dim_arg,
        &max_seq_len_arg,
        &mode_arg
    };

    printf("[*] Launching kernel: Grid=%u blocks, Block=%u threads...\n", grid_x, block_x);

    // Warmup run
    CHECK_CUDA(cuLaunchKernel(func,
                              grid_x, 1, 1,
                              block_x, 1, 1,
                              0, 0,
                              args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_Q_gpu, d_Q, q_bytes));
    CHECK_CUDA(cuMemcpyDtoH(h_K_cache_gpu, d_K_cache, cache_bytes));
    CHECK_CUDA(cuMemcpyDtoH(h_V_cache_gpu, d_V_cache, cache_bytes));

    // Validacion numerica estricta
    double max_diff_q = 0.0;
    double max_diff_k = 0.0;
    double max_diff_v = 0.0;
    size_t nonfinite_count = 0;
    size_t mismatch_count = 0;

    for (size_t i = 0; i < q_elements; i++) {
        if (!isfinite(h_Q_gpu[i])) nonfinite_count++;
        double diff = fabs((double)h_Q_gpu[i] - (double)h_Q_cpu[i]);
        if (diff > max_diff_q) max_diff_q = diff;
        if (diff > 1e-4) mismatch_count++;
    }

    size_t pos_offset_elements = (size_t)pos * head_dim;
    for (int h = 0; h < num_kv_heads; h++) {
        size_t off = ((size_t)h * max_seq_len + pos) * head_dim;
        for (int i = 0; i < head_dim; i++) {
            if (!isfinite(h_K_cache_gpu[off + i]) || !isfinite(h_V_cache_gpu[off + i])) nonfinite_count++;
            double diff_k = fabs((double)h_K_cache_gpu[off + i] - (double)h_K_cache_cpu[off + i]);
            double diff_v = fabs((double)h_V_cache_gpu[off + i] - (double)h_V_cache_cpu[off + i]);
            if (diff_k > max_diff_k) max_diff_k = diff_k;
            if (diff_v > max_diff_v) max_diff_v = diff_v;
            if (diff_k > 1e-4 || diff_v > 1e-4) mismatch_count++;
        }
    }

    printf("[*] Accuracy Assessment:\n");
    printf("    Non-finite Values  : %zu (Expected: 0)\n", nonfinite_count);
    printf("    Max Absolute Error : Q=%e, K_cache=%e, V_cache=%e\n",
           max_diff_q, max_diff_k, max_diff_v);
    printf("    Mismatches (> 1e-4): %zu\n", mismatch_count);

    if (nonfinite_count > 0 || mismatch_count > 0 || max_diff_q > 1e-4 || max_diff_k > 1e-4) {
        fprintf(stderr, "[-] FAILED: Numeric verification failed!\n");
        return 1;
    }
    printf("[+] NUMERIC VERIFICATION PASSED! GPU matches CPU within FP32 machine precision tolerance.\n\n");

    // Benchmark amortizado
    const int benchmark_iters = 100;
    printf("[*] Running benchmark (%d iterations with amortized GPU sync)...\n", benchmark_iters);

    CHECK_CUDA(cuCtxSynchronize());
    double t_start = get_time_sec();

    for (int iter = 0; iter < benchmark_iters; iter++) {
        CHECK_CUDA(cuLaunchKernel(func,
                                  grid_x, 1, 1,
                                  block_x, 1, 1,
                                  0, 0,
                                  args, NULL));
    }

    CHECK_CUDA(cuCtxSynchronize());
    double t_end = get_time_sec();

    double total_sec = t_end - t_start;
    double avg_ms = (total_sec / (double)benchmark_iters) * 1000.0;
    double tokens_per_sec = 1.0 / (avg_ms / 1000.0);

    printf("--------------------------------------------------------------------\n");
    printf("  PERFORMANCE METRICS (Fused RoPE + KV-Cache):\n");
    printf("  Total Time (%d runs) : %.3f ms\n", benchmark_iters, total_sec * 1000.0);
    printf("  Average Latency      : %.4f ms (%.1f µs per token RoPE+KV write)\n",
           avg_ms, avg_ms * 1000.0);
    printf("  Throughput           : %.1f tokens/sec\n", tokens_per_sec);
    printf("--------------------------------------------------------------------\n");

    cuMemFree(d_Q);
    cuMemFree(d_K_in);
    cuMemFree(d_V_in);
    cuMemFree(d_cos);
    cuMemFree(d_sin);
    cuMemFree(d_K_cache);
    cuMemFree(d_V_cache);
    free(h_Q_gpu);
    free(h_Q_cpu);
    free(h_K_in);
    free(h_V_in);
    free(h_cos);
    free(h_sin);
    free(h_K_cache_gpu);
    free(h_K_cache_cpu);
    free(h_V_cache_gpu);
    free(h_V_cache_cpu);
    cuFuncDestroy(func);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    printf("[+] Fused RoPE + KV-Cache benchmark finished cleanly with 0 errors.\n");
    return 0;
}
