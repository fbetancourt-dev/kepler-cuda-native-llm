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

// Referencia matemática pura en CPU de GEMV (y = A * x) con precisión double
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

int main(void) {
    printf("====================================================================\n");
    printf("  TRANSFORMER INFERENCE KERNEL BENCHMARK: GEMV (FP32 float4)\n");
    printf("  Vectorized 128-bit LD.128 Memory Loads + Kepler Warp SHFL\n");
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
    CHECK_CUDA(cuModuleLoad(&mod, "gemv_official.cubin"));
    printf("[+] Loaded CUBIN 'gemv_official.cubin' successfully.\n");

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "gemv_kernel"));
    printf("[+] Function 'gemv_kernel' resolved.\n");

    const int M = 2048; // Filas (hidden_dim salida)
    const int K = 2048; // Columnas (hidden_dim entrada)
    const int K4 = K / 4; // Elementos float4 por fila = 512
    const size_t matrix_elements = (size_t)M * K;
    const size_t matrix_bytes = matrix_elements * sizeof(float); // 16 MiB
    const size_t vec_in_bytes = (size_t)K * sizeof(float);       // 8 KiB
    const size_t vec_out_bytes = (size_t)M * sizeof(float);      // 8 KiB

    printf("[*] Layer Dimensions: W [%d x %d] (%zu MB), x [%d] (%zu KB), y [%d] (%zu KB)\n",
           M, K, matrix_bytes / (1024 * 1024), K, vec_in_bytes / 1024, M, vec_out_bytes / 1024);

    float *h_A = (float *)malloc(matrix_bytes);
    float *h_x = (float *)malloc(vec_in_bytes);
    float *h_y_gpu = (float *)malloc(vec_out_bytes);
    float *h_y_cpu = (float *)malloc(vec_out_bytes);

    // Inicializar datos deterministas
    for (size_t i = 0; i < matrix_elements; i++) {
        h_A[i] = ((float)(i % 1000) - 500.0f) * 0.001f;
    }
    for (int i = 0; i < K; i++) {
        h_x[i] = sinf((float)i * 0.05f);
    }

    printf("[*] Computing ground truth reference on CPU (double precision)...\n");
    cpu_gemv(h_y_cpu, h_A, h_x, M, K);

    CUdeviceptr d_y, d_A, d_x;
    CHECK_CUDA(cuMemAlloc(&d_y, vec_out_bytes));
    CHECK_CUDA(cuMemAlloc(&d_A, matrix_bytes));
    CHECK_CUDA(cuMemAlloc(&d_x, vec_in_bytes));

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, matrix_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_x, h_x, vec_in_bytes));

    const unsigned int block_x = 32;
    const unsigned int block_y = 4; // 4 warps por bloque = 128 hilos
    const unsigned int grid_x = (M + block_y - 1) / block_y; // 512 bloques
    int M_arg = M;
    int K4_arg = K4;

    void *args[] = {
        &d_y,
        &d_A,
        &d_x,
        &M_arg,
        &K4_arg
    };

    printf("[*] Launching kernel: Grid=%u blocks, Block=(%u, %u) [128 threads]...\n",
           grid_x, block_x, block_y);

    // Warmup run
    CHECK_CUDA(cuLaunchKernel(func,
                              grid_x, 1, 1,
                              block_x, block_y, 1,
                              0, 0,
                              args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_y_gpu, d_y, vec_out_bytes));

    // Validacion numerica estricta contra NaNs, Infs y desviaciones
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t nonfinite_count = 0;
    size_t mismatch_count = 0;

    for (int i = 0; i < M; i++) {
        float gpu_val = h_y_gpu[i];
        float cpu_val = h_y_cpu[i];

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
                printf("[-] Mismatch at row %d: GPU=%f, CPU=%f, diff=%e\n",
                       i, gpu_val, cpu_val, diff);
            }
        }
    }

    printf("[*] Accuracy Assessment:\n");
    printf("    Non-finite Values  : %zu (Expected: 0)\n", nonfinite_count);
    printf("    Max Absolute Error : %e\n", max_abs_diff);
    printf("    Max Relative Error : %e\n", max_rel_diff);
    printf("    Mismatches (> 5e-3): %zu / %d\n", mismatch_count, M);

    if (nonfinite_count > 0 || mismatch_count > 0 || max_abs_diff > 1e-2) {
        fprintf(stderr, "[-] FAILED: Numeric verification failed!\n");
        return 1;
    }
    printf("[+] NUMERIC VERIFICATION PASSED! GPU matches CPU within FP32 machine precision tolerance.\n\n");

    // Protocolo de medicion riguroso: sincronizacion GPU antes y despues del loop
    const int benchmark_iters = 50;
    printf("[*] Running benchmark (%d iterations with amortized GPU sync)...\n", benchmark_iters);

    CHECK_CUDA(cuCtxSynchronize());
    double t_start = get_time_sec();

    for (int iter = 0; iter < benchmark_iters; iter++) {
        CHECK_CUDA(cuLaunchKernel(func,
                                  grid_x, 1, 1,
                                  block_x, block_y, 1,
                                  0, 0,
                                  args, NULL));
    }

    CHECK_CUDA(cuCtxSynchronize());
    double t_end = get_time_sec();

    double total_sec = t_end - t_start;
    double avg_ms = (total_sec / (double)benchmark_iters) * 1000.0;
    double gflops = (2.0 * (double)M * (double)K / 1e9) / (avg_ms / 1000.0);
    double total_bytes = (double)(matrix_bytes + vec_in_bytes + vec_out_bytes);
    double bandwidth_gb_s = (total_bytes / (avg_ms / 1000.0)) / 1e9;
    double tokens_per_sec = 1.0 / (avg_ms / 1000.0);

    printf("--------------------------------------------------------------------\n");
    printf("  RIGOROUS PERFORMANCE METRICS (Amortized End-to-End):\n");
    printf("  Total Time (%d runs) : %.3f ms\n", benchmark_iters, total_sec * 1000.0);
    printf("  Average Latency      : %.4f ms (%.1f µs per token projection)\n",
           avg_ms, avg_ms * 1000.0);
    printf("  Computational Speed  : %.2f GFLOPS\n", gflops);
    printf("  Effective Bandwidth  : %.2f GB/s\n", bandwidth_gb_s);
    printf("  Max Decode Rate      : %.1f single-layer projections/sec\n", tokens_per_sec);
    printf("--------------------------------------------------------------------\n");

    cuMemFree(d_y);
    cuMemFree(d_A);
    cuMemFree(d_x);
    free(h_A);
    free(h_x);
    free(h_y_gpu);
    free(h_y_cpu);
    cuFuncDestroy(func);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    printf("[+] GEMV float4 benchmark finished cleanly with 0 errors.\n");
    return 0;
}
