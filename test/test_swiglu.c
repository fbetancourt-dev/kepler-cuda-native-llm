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

// Referencia en CPU de SwiGLU con precision double
static void cpu_swiglu(float *out, const float *gate, const float *up, int dim) {
    for (int i = 0; i < dim; i++) {
        double g = (double)gate[i];
        double u = (double)up[i];
        double silu = g / (1.0 + exp(-g));
        out[i] = (float)(silu * u);
    }
}

// Referencia en CPU de Residual Add
static void cpu_residual_add(float *x, const float *residual, int dim) {
    for (int i = 0; i < dim; i++) {
        x[i] += residual[i];
    }
}

int main(void) {
    printf("====================================================================\n");
    printf("  TRANSFORMER INFERENCE KERNEL BENCHMARK: SwiGLU & RESIDUAL ADD\n");
    printf("  LLaMA / Mistral MLP Activation + Element-wise Skip Connection\n");
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
    CHECK_CUDA(cuModuleLoad(&mod, "swiglu_official.cubin"));
    printf("[+] Loaded CUBIN 'swiglu_official.cubin' successfully.\n");

    CUfunction func_swiglu, func_residual;
    CHECK_CUDA(cuModuleGetFunction(&func_swiglu, mod, "swiglu_kernel"));
    CHECK_CUDA(cuModuleGetFunction(&func_residual, mod, "residual_add_kernel"));
    printf("[+] Functions 'swiglu_kernel' and 'residual_add_kernel' resolved.\n");

    // Dimension intermedia MLP de TinyLlama-1.1B
    const int intermediate_dim = 5632;
    const size_t tensor_bytes = (size_t)intermediate_dim * sizeof(float);

    printf("[*] MLP Intermediate Dimension: %d floats (%zu KB)\n",
           intermediate_dim, tensor_bytes / 1024);

    float *h_gate = (float *)malloc(tensor_bytes);
    float *h_up = (float *)malloc(tensor_bytes);
    float *h_out_gpu = (float *)malloc(tensor_bytes);
    float *h_out_cpu = (float *)malloc(tensor_bytes);
    float *h_residual = (float *)malloc(tensor_bytes);

    // Inicializar datos con rango [-5.0, 5.0] para probar saturacion y linealidad de SiLU
    for (int i = 0; i < intermediate_dim; i++) {
        h_gate[i] = ((float)(i % 1000) - 500.0f) * 0.01f;
        h_up[i] = sinf((float)i * 0.017f);
        h_residual[i] = cosf((float)i * 0.023f) * 0.5f;
    }

    printf("[*] Computing ground truth reference on CPU (double precision)...\n");
    cpu_swiglu(h_out_cpu, h_gate, h_up, intermediate_dim);

    CUdeviceptr d_out, d_gate, d_up, d_residual;
    CHECK_CUDA(cuMemAlloc(&d_out, tensor_bytes));
    CHECK_CUDA(cuMemAlloc(&d_gate, tensor_bytes));
    CHECK_CUDA(cuMemAlloc(&d_up, tensor_bytes));
    CHECK_CUDA(cuMemAlloc(&d_residual, tensor_bytes));

    CHECK_CUDA(cuMemcpyHtoD(d_gate, h_gate, tensor_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_up, h_up, tensor_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_residual, h_residual, tensor_bytes));

    const unsigned int threads_per_block = 256;
    const unsigned int num_blocks = (intermediate_dim + threads_per_block - 1) / threads_per_block;
    int dim_arg = intermediate_dim;

    void *args_swiglu[] = {
        &d_out,
        &d_gate,
        &d_up,
        &dim_arg
    };

    printf("[*] Launching SwiGLU: Grid=%u blocks, Block=%u threads...\n",
           num_blocks, threads_per_block);

    // Warmup
    CHECK_CUDA(cuLaunchKernel(func_swiglu,
                              num_blocks, 1, 1,
                              threads_per_block, 1, 1,
                              0, 0,
                              args_swiglu, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_out_gpu, d_out, tensor_bytes));

    // Validacion numerica de SwiGLU
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t nonfinite_count = 0;
    size_t mismatch_count = 0;

    for (int i = 0; i < intermediate_dim; i++) {
        float gpu_val = h_out_gpu[i];
        float cpu_val = h_out_cpu[i];

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
                printf("[-] Mismatch at idx %d: GPU=%f, CPU=%f, diff=%e\n",
                       i, gpu_val, cpu_val, diff);
            }
        }
    }

    printf("[*] SwiGLU Accuracy Assessment:\n");
    printf("    Non-finite Values  : %zu (Expected: 0)\n", nonfinite_count);
    printf("    Max Absolute Error : %e\n", max_abs_diff);
    printf("    Max Relative Error : %e\n", max_rel_diff);
    printf("    Mismatches (> 1e-4): %zu / %d\n", mismatch_count, intermediate_dim);

    if (nonfinite_count > 0 || mismatch_count > 0 || max_abs_diff > 1e-3) {
        fprintf(stderr, "[-] FAILED: SwiGLU numeric verification failed!\n");
        return 1;
    }
    printf("[+] SWIGLU VERIFICATION PASSED! GPU matches CPU within FP32 machine precision tolerance.\n");

    // Validar Residual Add (d_out += d_residual)
    void *args_residual[] = {
        &d_out,
        &d_residual,
        &dim_arg
    };
    CHECK_CUDA(cuLaunchKernel(func_residual,
                              num_blocks, 1, 1,
                              threads_per_block, 1, 1,
                              0, 0,
                              args_residual, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_out_gpu, d_out, tensor_bytes));
    cpu_residual_add(h_out_cpu, h_residual, intermediate_dim);

    double max_diff_res = 0.0;
    for (int i = 0; i < intermediate_dim; i++) {
        double diff = fabs((double)h_out_gpu[i] - (double)h_out_cpu[i]);
        if (diff > max_diff_res) max_diff_res = diff;
    }
    printf("[*] Residual Add Max Absolute Error: %e\n", max_diff_res);
    if (max_diff_res > 1e-5) {
        fprintf(stderr, "[-] FAILED: Residual Add numeric verification failed!\n");
        return 1;
    }
    printf("[+] RESIDUAL ADD VERIFICATION PASSED!\n\n");

    // Benchmark amortizado
    const int benchmark_iters = 100;
    printf("[*] Running benchmark (%d iterations with amortized GPU sync)...\n", benchmark_iters);

    CHECK_CUDA(cuCtxSynchronize());
    double t_start = get_time_sec();

    for (int iter = 0; iter < benchmark_iters; iter++) {
        CHECK_CUDA(cuLaunchKernel(func_swiglu,
                                  num_blocks, 1, 1,
                                  threads_per_block, 1, 1,
                                  0, 0,
                                  args_swiglu, NULL));
        CHECK_CUDA(cuLaunchKernel(func_residual,
                                  num_blocks, 1, 1,
                                  threads_per_block, 1, 1,
                                  0, 0,
                                  args_residual, NULL));
    }

    CHECK_CUDA(cuCtxSynchronize());
    double t_end = get_time_sec();

    double total_sec = t_end - t_start;
    double avg_ms = (total_sec / (double)benchmark_iters) * 1000.0;
    double tokens_per_sec = 1.0 / (avg_ms / 1000.0);

    printf("--------------------------------------------------------------------\n");
    printf("  PERFORMANCE METRICS (SwiGLU + Residual Add):\n");
    printf("  Total Time (%d runs) : %.3f ms\n", benchmark_iters, total_sec * 1000.0);
    printf("  Average Latency      : %.4f ms (%.1f µs per MLP activation + skip)\n",
           avg_ms, avg_ms * 1000.0);
    printf("  Throughput           : %.1f activations/sec\n", tokens_per_sec);
    printf("--------------------------------------------------------------------\n");

    cuMemFree(d_out);
    cuMemFree(d_gate);
    cuMemFree(d_up);
    cuMemFree(d_residual);
    free(h_gate);
    free(h_up);
    free(h_out_gpu);
    free(h_out_cpu);
    free(h_residual);
    cuFuncDestroy(func_swiglu);
    cuFuncDestroy(func_residual);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    printf("[+] SwiGLU & Residual Add benchmark finished cleanly with 0 errors.\n");
    return 0;
}
