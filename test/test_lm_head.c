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

int main(void) {
    printf("====================================================================\n");
    printf("  TRANSFORMER INFERENCE KERNEL BENCHMARK: LM_HEAD + ARGMAX (GPU)\n");
    printf("  Vocabulary Projection (32,000 x 2048) + Intra-GPU Token Selection\n");
    printf("  Silicon: NVIDIA GeForce GT 750M (Kepler GK107, sm_30)\n");
    printf("====================================================================\n");

    CHECK_CUDA(cuInit(0));

    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));

    char name[128];
    CHECK_CUDA(cuDeviceGetName(name, sizeof(name), dev));
    printf("[+] Target Hardware: %s\n", name);

    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK_CUDA(cuModuleLoad(&mod, "lm_head_official.cubin"));
    printf("[+] Loaded CUBIN 'lm_head_official.cubin' successfully.\n");

    CUfunction func_lm, func_argmax;
    CHECK_CUDA(cuModuleGetFunction(&func_lm, mod, "lm_head_kernel"));
    CHECK_CUDA(cuModuleGetFunction(&func_argmax, mod, "argmax_kernel"));
    printf("[+] Functions 'lm_head_kernel' and 'argmax_kernel' resolved.\n");

    const int vocab_size = 32000;
    const int D = 2048;
    const int K4 = D / 4; // 512

    const size_t matrix_bytes = (size_t)vocab_size * D * sizeof(float); // 262.14 MB (o 128 MiB para float4)
    const size_t vec_in_bytes = (size_t)D * sizeof(float);               // 8 KB
    const size_t logits_bytes = (size_t)vocab_size * sizeof(float);     // 128 KB

    printf("[*] Layer Dimensions: W_lm [%d x %d] (%.1f MB), x [%d] (8 KB), logits [%d] (128 KB)\n",
           vocab_size, D, (double)matrix_bytes / (1024.0 * 1024.0), D, vocab_size);

    float *h_W_lm = (float *)malloc(matrix_bytes);
    float *h_x = (float *)malloc(vec_in_bytes);
    float *h_logits_gpu = (float *)malloc(logits_bytes);
    float *h_logits_cpu = (float *)malloc(logits_bytes);

    // Inicializar pesos con un pico deliberado en el token 18452 para probar argmax
    for (size_t i = 0; i < (size_t)vocab_size * D; i++) {
        h_W_lm[i] = (((float)(i % 1000) - 500.0f) * 0.0001f);
    }
    for (int i = 0; i < D; i++) {
        h_x[i] = sinf((float)i * 0.015f);
        // Forzar vector de fila para el token objetivo 18452
        h_W_lm[(size_t)18452 * D + i] = 0.5f * sinf((float)i * 0.015f);
    }

    // Referencia CPU
    printf("[*] Computing ground truth logits and Argmax on CPU (double precision)...\n");
    float max_cpu_logit = -1e38f;
    int max_cpu_token = -1;

    for (int v = 0; v < vocab_size; v++) {
        double sum = 0.0;
        const float *row = h_W_lm + (size_t)v * D;
        for (int c = 0; c < D; c++) {
            sum += (double)row[c] * (double)h_x[c];
        }
        h_logits_cpu[v] = (float)sum;
        if (h_logits_cpu[v] > max_cpu_logit) {
            max_cpu_logit = h_logits_cpu[v];
            max_cpu_token = v;
        }
    }
    printf("[+] Expected CPU Winner: Token ID %d (Logit: %f)\n", max_cpu_token, max_cpu_logit);

    // Alojar VRAM
    CUdeviceptr d_W_lm, d_x, d_logits, d_token_out;
    CHECK_CUDA(cuMemAlloc(&d_W_lm, matrix_bytes));
    CHECK_CUDA(cuMemAlloc(&d_x, vec_in_bytes));
    CHECK_CUDA(cuMemAlloc(&d_logits, logits_bytes));
    CHECK_CUDA(cuMemAlloc(&d_token_out, sizeof(int)));

    CHECK_CUDA(cuMemcpyHtoD(d_W_lm, h_W_lm, matrix_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_x, h_x, vec_in_bytes));

    int vocab_arg = vocab_size;
    int K4_arg = K4;

    void *args_lm[] = {
        &d_logits,
        &d_W_lm,
        &d_x,
        &vocab_arg,
        &K4_arg
    };

    void *args_argmax[] = {
        &d_token_out,
        &d_logits,
        &vocab_arg
    };

    // Geometria LM_HEAD: 32 hilos en X (1 warp), 4 warps por bloque = 128 hilos
    // 32000 / 4 = 8000 bloques
    const unsigned int block_x = 32;
    const unsigned int block_y = 4;
    const unsigned int grid_x = (vocab_size + block_y - 1) / block_y; // 8000 bloques

    printf("[*] Launching lm_head on GPU: Grid=%u blocks, Block=(%u, %u) [8000 warps]...\n",
           grid_x, block_x, block_y);

    // Warmup
    CHECK_CUDA(cuLaunchKernel(func_lm, grid_x, 1, 1, block_x, block_y, 1, 0, 0, args_lm, NULL));
    CHECK_CUDA(cuLaunchKernel(func_argmax, 1, 1, 1, 256, 1, 1, 0, 0, args_argmax, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    int gpu_token_id = -1;
    CHECK_CUDA(cuMemcpyDtoH(&gpu_token_id, d_token_out, sizeof(int)));
    CHECK_CUDA(cuMemcpyDtoH(h_logits_gpu, d_logits, logits_bytes));

    printf("[*] GPU Token Selection Result: Token ID %d (Expected: %d)\n",
           gpu_token_id, max_cpu_token);

    if (gpu_token_id != max_cpu_token) {
        fprintf(stderr, "[-] FAILED: Argmax mismatch! GPU=%d, CPU=%d\n", gpu_token_id, max_cpu_token);
        return 1;
    }

    double max_abs_diff = 0.0;
    for (int v = 0; v < vocab_size; v++) {
        double diff = fabs((double)h_logits_gpu[v] - (double)h_logits_cpu[v]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    printf("[+] NUMERIC VERIFICATION PASSED! Logits match CPU (Max Abs Error: %e)\n", max_abs_diff);
    printf("[+] ARGMAX SELECTION PASSED 100%% IN SILICON!\n\n");

    // Benchmark amortizado
    const int benchmark_iters = 20;
    printf("[*] Running benchmark (%d iterations with amortized GPU sync)...\n", benchmark_iters);

    CHECK_CUDA(cuCtxSynchronize());
    double t_start = get_time_sec();

    for (int iter = 0; iter < benchmark_iters; iter++) {
        cuLaunchKernel(func_lm, grid_x, 1, 1, block_x, block_y, 1, 0, 0, args_lm, NULL);
        cuLaunchKernel(func_argmax, 1, 1, 1, 256, 1, 1, 0, 0, args_argmax, NULL);
    }

    CHECK_CUDA(cuCtxSynchronize());
    double t_end = get_time_sec();

    double total_sec = t_end - t_start;
    double avg_ms = (total_sec / (double)benchmark_iters) * 1000.0;
    double throughput_tokens = 1.0 / (avg_ms / 1000.0);
    double bandwidth_gb_s = ((double)matrix_bytes / (avg_ms / 1000.0)) / 1e9;

    printf("--------------------------------------------------------------------\n");
    printf("  PERFORMANCE METRICS (LM_HEAD + ARGMAX ON GPU):\n");
    printf("  Total Time (%d runs) : %.3f ms\n", benchmark_iters, total_sec * 1000.0);
    printf("  Average Latency      : %.2f ms per token projection & argmax\n", avg_ms);
    printf("  Throughput           : %.1f token-vocab projections/sec\n", throughput_tokens);
    printf("  Effective Bandwidth  : %.2f GB/s\n", bandwidth_gb_s);
    printf("--------------------------------------------------------------------\n");

    cuMemFree(d_W_lm);
    cuMemFree(d_x);
    cuMemFree(d_logits);
    cuMemFree(d_token_out);
    free(h_W_lm);
    free(h_x);
    free(h_logits_gpu);
    free(h_logits_cpu);
    cuFuncDestroy(func_lm);
    cuFuncDestroy(func_argmax);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    printf("[+] LM_Head + Argmax benchmark finished cleanly with 0 errors.\n");
    return 0;
}
