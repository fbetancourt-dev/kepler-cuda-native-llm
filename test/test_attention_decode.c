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

// Referencia en CPU de Attention Decode con Online Softmax en doble precision
static void cpu_attention_decode(
    float *out, const float *Q, const float *K_cache, const float *V_cache,
    int num_heads, int num_kv_heads, int head_dim, int max_seq_len, int context_len, float scale)
{
    int gqa_ratio = num_heads / num_kv_heads;

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / gqa_ratio;
        const float *q_head = Q + (size_t)h * head_dim;
        float *out_head = out + (size_t)h * head_dim;

        double max_score = -1e30;
        double sum_exp = 0.0;
        double acc[64] = {0.0};

        for (int t = 0; t < context_len; t++) {
            size_t cache_off = ((size_t)kv_h * max_seq_len + t) * head_dim;
            const float *k_vec = K_cache + cache_off;
            const float *v_vec = V_cache + cache_off;

            double dot = 0.0;
            for (int i = 0; i < head_dim; i++) {
                dot += (double)q_head[i] * (double)k_vec[i];
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
            out_head[i] = (float)(acc[i] * inv_sum);
        }
    }
}

int main(void) {
    printf("====================================================================\n");
    printf("  TRANSFORMER INFERENCE KERNEL BENCHMARK: ONLINE ATTENTION DECODE\n");
    printf("  FlashAttention Single-Pass Online Softmax + GQA (Kepler sm_30)\n");
    printf("  Silicon: NVIDIA GeForce GT 750M (Kepler GK107)\n");
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
    CHECK_CUDA(cuModuleLoad(&mod, "attention_decode_official.cubin"));
    printf("[+] Loaded CUBIN 'attention_decode_official.cubin' successfully.\n");

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "attention_decode_kernel"));
    printf("[+] Function 'attention_decode_kernel' resolved.\n");

    const int num_heads = 32;
    const int num_kv_heads = 4;
    const int head_dim = 64;
    const int max_seq_len = 512;
    const int context_len = 128; // Secuencia acumulada previa
    const float scale = 1.0f / sqrtf((float)head_dim);

    const size_t q_elements = (size_t)num_heads * head_dim; // 2048 floats
    const size_t cache_elements = (size_t)num_kv_heads * max_seq_len * head_dim; // 131,072 floats
    const size_t out_elements = q_elements;

    const size_t q_bytes = q_elements * sizeof(float);
    const size_t cache_bytes = cache_elements * sizeof(float);
    const size_t out_bytes = out_elements * sizeof(float);

    printf("[*] Configuration: Q-Heads=%d, KV-Heads=%d (GQA 8:1), HeadDim=%d\n",
           num_heads, num_kv_heads, head_dim);
    printf("[*] Attending across Context Length: %d tokens in KV-Cache\n", context_len);

    float *h_Q = (float *)malloc(q_bytes);
    float *h_K_cache = (float *)malloc(cache_bytes);
    float *h_V_cache = (float *)malloc(cache_bytes);
    float *h_out_gpu = (float *)malloc(out_bytes);
    float *h_out_cpu = (float *)malloc(out_bytes);

    // Inicializar datos
    for (size_t i = 0; i < q_elements; i++) {
        h_Q[i] = sinf((float)i * 0.02f) * 0.5f;
    }
    for (size_t i = 0; i < cache_elements; i++) {
        h_K_cache[i] = cosf((float)i * 0.03f) * 0.5f;
        h_V_cache[i] = sinf((float)i * 0.04f) * 0.5f;
    }

    printf("[*] Computing ground truth reference on CPU (double precision)...\n");
    cpu_attention_decode(h_out_cpu, h_Q, h_K_cache, h_V_cache,
                         num_heads, num_kv_heads, head_dim, max_seq_len, context_len, scale);

    CUdeviceptr d_out, d_Q, d_K_cache, d_V_cache;
    CHECK_CUDA(cuMemAlloc(&d_out, out_bytes));
    CHECK_CUDA(cuMemAlloc(&d_Q, q_bytes));
    CHECK_CUDA(cuMemAlloc(&d_K_cache, cache_bytes));
    CHECK_CUDA(cuMemAlloc(&d_V_cache, cache_bytes));

    CHECK_CUDA(cuMemcpyHtoD(d_Q, h_Q, q_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_K_cache, h_K_cache, cache_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_V_cache, h_V_cache, cache_bytes));

    // Geometria: 32 bloques (1 por cabeza Q), 32 hilos por bloque (1 warp)
    const unsigned int grid_x = num_heads;
    const unsigned int block_x = 32;
    int num_heads_arg = num_heads;
    int num_kv_heads_arg = num_kv_heads;
    int head_dim_arg = head_dim;
    int max_seq_len_arg = max_seq_len;
    int context_len_arg = context_len;
    float scale_arg = scale;

    void *args[] = {
        &d_out,
        &d_Q,
        &d_K_cache,
        &d_V_cache,
        &num_heads_arg,
        &num_kv_heads_arg,
        &head_dim_arg,
        &max_seq_len_arg,
        &context_len_arg,
        &scale_arg
    };

    printf("[*] Launching kernel: Grid=%u blocks, Block=%u threads...\n", grid_x, block_x);

    // Warmup run
    CHECK_CUDA(cuLaunchKernel(func,
                              grid_x, 1, 1,
                              block_x, 1, 1,
                              0, 0,
                              args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_out_gpu, d_out, out_bytes));

    // Validacion numerica estricta
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t nonfinite_count = 0;
    size_t mismatch_count = 0;

    for (size_t i = 0; i < out_elements; i++) {
        float gpu_val = h_out_gpu[i];
        float cpu_val = h_out_cpu[i];

        if (!isfinite(gpu_val) || !isfinite(cpu_val)) {
            nonfinite_count++;
            continue;
        }

        double diff = fabs((double)gpu_val - (double)cpu_val);
        double ref = fabs((double)cpu_val);
        double rel = ref > 1e-5 ? diff / ref : diff;

        if (diff > max_abs_diff) max_abs_diff = diff;
        if (rel > max_rel_diff) max_rel_diff = rel;

        if (diff > 5e-4) {
            mismatch_count++;
            if (mismatch_count <= 5) {
                printf("[-] Mismatch at idx %zu: GPU=%f, CPU=%f, diff=%e\n",
                       i, gpu_val, cpu_val, diff);
            }
        }
    }

    printf("[*] Accuracy Assessment:\n");
    printf("    Non-finite Values  : %zu (Expected: 0)\n", nonfinite_count);
    printf("    Max Absolute Error : %e\n", max_abs_diff);
    printf("    Max Relative Error : %e\n", max_rel_diff);
    printf("    Mismatches (> 5e-4): %zu / %zu\n", mismatch_count, out_elements);

    if (nonfinite_count > 0 || mismatch_count > 0 || max_abs_diff > 1e-3) {
        fprintf(stderr, "[-] FAILED: Numeric verification failed!\n");
        return 1;
    }
    printf("[+] NUMERIC VERIFICATION PASSED! GPU matches CPU within FP32 machine precision tolerance.\n\n");

    // Benchmark amortizado (100 iteraciones cronometradas)
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
    printf("  PERFORMANCE METRICS (Online Attention Decode):\n");
    printf("  Total Time (%d runs) : %.3f ms\n", benchmark_iters, total_sec * 1000.0);
    printf("  Average Latency      : %.4f ms (%.1f µs per token across %d context tokens)\n",
           avg_ms, avg_ms * 1000.0, context_len);
    printf("  Throughput           : %.1f decode tokens/sec\n", tokens_per_sec);
    printf("--------------------------------------------------------------------\n");

    cuMemFree(d_out);
    cuMemFree(d_Q);
    cuMemFree(d_K_cache);
    cuMemFree(d_V_cache);
    free(h_Q);
    free(h_K_cache);
    free(h_V_cache);
    free(h_out_gpu);
    free(h_out_cpu);
    cuFuncDestroy(func);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    printf("[+] Attention Decode benchmark finished cleanly with 0 errors.\n");
    return 0;
}
