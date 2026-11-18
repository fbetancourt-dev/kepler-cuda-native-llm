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

// Referencia en CPU de doble precision pura para RMSNorm
static void cpu_rms_norm(float *y, const float *x, const float *w, int num_tokens, int hidden_dim, float eps) {
    for (int t = 0; t < num_tokens; t++) {
        const float *x_tok = x + (size_t)t * hidden_dim;
        float *y_tok = y + (size_t)t * hidden_dim;

        double sum_sq = 0.0;
        for (int i = 0; i < hidden_dim; i++) {
            double v = (double)x_tok[i];
            sum_sq += v * v;
        }
        double mean_sq = sum_sq / (double)hidden_dim;
        double rrms = 1.0 / sqrt(mean_sq + (double)eps);

        for (int i = 0; i < hidden_dim; i++) {
            y_tok[i] = (float)((double)x_tok[i] * rrms * (double)w[i]);
        }
    }
}

int main(void) {
    printf("====================================================================\n");
    printf("  TRANSFORMER INFERENCE KERNEL BENCHMARK: RMSNorm (FP32)\n");
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
    CHECK_CUDA(cuModuleLoad(&mod, "rms_norm_official.cubin"));
    printf("[+] Loaded CUBIN 'rms_norm_official.cubin' successfully.\n");

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "rms_norm_kernel"));
    printf("[+] Function 'rms_norm_kernel' resolved.\n");

    const int num_tokens = 64;
    const int hidden_dim = 2048;
    const float eps = 1e-5f;
    const size_t total_elements = (size_t)num_tokens * hidden_dim;
    const size_t tensor_bytes = total_elements * sizeof(float);
    const size_t weight_bytes = hidden_dim * sizeof(float);

    printf("[*] Model Dimensions: tokens=%d, hidden_dim=%d, total_floats=%zu (%zu KB)\n",
           num_tokens, hidden_dim, total_elements, tensor_bytes / 1024);

    float *h_x = (float *)malloc(tensor_bytes);
    float *h_w = (float *)malloc(weight_bytes);
    float *h_y_gpu = (float *)malloc(tensor_bytes);
    float *h_y_cpu = (float *)malloc(tensor_bytes);

    for (size_t i = 0; i < total_elements; i++) {
        h_x[i] = sinf((float)i * 0.013f) * 1.5f + 0.1f * cosf((float)i * 0.007f);
    }
    for (int i = 0; i < hidden_dim; i++) {
        h_w[i] = 1.0f + 0.2f * sinf((float)i * 0.05f);
    }

    printf("[*] Computing ground truth reference on CPU (pure double precision)...\n");
    cpu_rms_norm(h_y_cpu, h_x, h_w, num_tokens, hidden_dim, eps);

    CUdeviceptr d_x, d_w, d_y;
    CHECK_CUDA(cuMemAlloc(&d_x, tensor_bytes));
    CHECK_CUDA(cuMemAlloc(&d_w, weight_bytes));
    CHECK_CUDA(cuMemAlloc(&d_y, tensor_bytes));

    CHECK_CUDA(cuMemcpyHtoD(d_x, h_x, tensor_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_w, h_w, weight_bytes));

    const unsigned int threads_per_block = 256;
    const unsigned int dynamic_smem = threads_per_block * sizeof(float); // 1024 B
    int hidden_dim_arg = hidden_dim;
    float eps_arg = eps;

    void *args[] = {
        &d_y,
        &d_x,
        &d_w,
        &hidden_dim_arg,
        &eps_arg
    };

    // Warmup run
    CHECK_CUDA(cuLaunchKernel(func,
                              num_tokens, 1, 1,
                              threads_per_block, 1, 1,
                              dynamic_smem, 0,
                              args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_y_gpu, d_y, tensor_bytes));

    // Validacion numerica rigurosa
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t nonfinite_count = 0;
    size_t mismatch_count = 0;

    for (size_t i = 0; i < total_elements; i++) {
        float gpu_val = h_y_gpu[i];
        float cpu_val = h_y_cpu[i];

        if (!isfinite(gpu_val) || !isfinite(cpu_val)) {
            nonfinite_count++;
            continue;
        }

        double diff = fabs((double)gpu_val - (double)cpu_val);
        double ref = fabs((double)cpu_val);
        double rel = ref > 1e-6 ? diff / ref : diff;

        if (diff > max_abs_diff) max_abs_diff = diff;
        if (rel > max_rel_diff) max_rel_diff = rel;

        if (diff > 1e-4) {
            mismatch_count++;
            if (mismatch_count <= 5) {
                int tok = (int)(i / hidden_dim);
                int dim = (int)(i % hidden_dim);
                printf("[-] Mismatch at token %d, dim %d: GPU=%f, CPU=%f, diff=%e\n",
                       tok, dim, gpu_val, cpu_val, diff);
            }
        }
    }

    printf("[*] Accuracy Assessment:\n");
    printf("    Non-finite Values  : %zu (Expected: 0)\n", nonfinite_count);
    printf("    Max Absolute Error : %e\n", max_abs_diff);
    printf("    Max Relative Error : %e\n", max_rel_diff);
    printf("    Mismatches (> 1e-4): %zu / %zu\n", mismatch_count, total_elements);

    if (nonfinite_count > 0 || mismatch_count > 0 || max_abs_diff > 1e-3) {
        fprintf(stderr, "[-] FAILED: Numeric verification failed!\n");
        return 1;
    }
    printf("[+] NUMERIC VERIFICATION PASSED! GPU matches CPU within FP32 machine precision tolerance.\n\n");

    const int benchmark_iters = 50;
    printf("[*] Running benchmark (%d iterations with amortized GPU sync)...\n", benchmark_iters);

    CHECK_CUDA(cuCtxSynchronize());
    double t_start = get_time_sec();

    for (int iter = 0; iter < benchmark_iters; iter++) {
        CHECK_CUDA(cuLaunchKernel(func,
                                  num_tokens, 1, 1,
                                  threads_per_block, 1, 1,
                                  dynamic_smem, 0,
                                  args, NULL));
    }

    CHECK_CUDA(cuCtxSynchronize());
    double t_end = get_time_sec();

    double total_sec = t_end - t_start;
    double avg_ms = (total_sec / (double)benchmark_iters) * 1000.0;
    double bytes_per_iter = (double)(tensor_bytes + weight_bytes + tensor_bytes);
    double bandwidth_gb_s = (bytes_per_iter / (avg_ms / 1000.0)) / 1e9;
    double batched_tokens_per_sec = (double)num_tokens / (avg_ms / 1000.0);

    printf("--------------------------------------------------------------------\n");
    printf("  RIGOROUS PERFORMANCE METRICS (Amortized End-to-End):\n");
    printf("  Total Time (%d runs) : %.3f ms\n", benchmark_iters, total_sec * 1000.0);
    printf("  Average Latency      : %.4f ms (%.1f µs per batch of %d tokens)\n",
           avg_ms, avg_ms * 1000.0, num_tokens);
    printf("  Batched Throughput   : %.1f tokens/sec\n", batched_tokens_per_sec);
    printf("  Useful Bandwidth     : %.2f GB/s\n", bandwidth_gb_s);
    printf("--------------------------------------------------------------------\n");

    cuMemFree(d_x);
    cuMemFree(d_w);
    cuMemFree(d_y);
    free(h_x);
    free(h_w);
    free(h_y_gpu);
    free(h_y_cpu);
    cuFuncDestroy(func);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    printf("[+] RMSNorm benchmark finished cleanly with 0 errors.\n");
    return 0;
}
