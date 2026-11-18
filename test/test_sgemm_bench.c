#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "[-] CUDA Error in %s: code %d\n", #call, err); \
        return 1; \
    } \
} while (0)

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("============================================================\n");
    printf("  TEST PASO 4: SGEMM SATURATION BENCHMARK (1024x1024)\n");
    printf("  NVIDIA GeForce GT 750M SM_30 (384 Cores, 926 MHz Turbo)\n");
    printf("  Tiled Shared Memory GEMM: C = A * B (2.15 GFLOP per pass)\n");
    printf("============================================================\n");

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    const char *cubin_path = "/home/fbetancourt/Gemini/Sandbox/cuda_kepler_native/sgemm_official.cubin";
    printf("[*] Loading SGEMM CUBIN from %s...\n", cubin_path);

    CUmodule mod;
    CHECK_CUDA(cuModuleLoad(&mod, cubin_path));

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "sgemm_tiled_kernel"));
    printf("[+] Kernel 'sgemm_tiled_kernel' loaded successfully!\n");

    const int N = 1024;
    const size_t num_elements = (size_t)N * N;
    const size_t matrix_bytes = num_elements * sizeof(float); // 4 MiB per matrix
    const double gflops_per_iter = (2.0 * (double)N * (double)N * (double)N) * 1e-9; // 2.147483648 GFLOP

    printf("[*] Allocating 3x %zu MB matrices in GPU VRAM (Total %zu MB)...\n",
           matrix_bytes / (1024 * 1024), (3 * matrix_bytes) / (1024 * 1024));

    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, matrix_bytes));
    CHECK_CUDA(cuMemAlloc(&d_B, matrix_bytes));
    CHECK_CUDA(cuMemAlloc(&d_C, matrix_bytes));

    float *h_A = (float *)malloc(matrix_bytes);
    float *h_B = (float *)malloc(matrix_bytes);
    float *h_C = (float *)malloc(matrix_bytes);

    for (size_t i = 0; i < num_elements; i++) {
        h_A[i] = 1.0f; // A = all 1.0
        h_B[i] = 2.0f; // B = all 2.0 -> Expected C[i][j] = sum_{k=0..1023}(1.0 * 2.0) = 2048.0
        h_C[i] = 0.0f;
    }

    printf("[*] Copying input matrices A and B to GPU VRAM...\n");
    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, matrix_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, matrix_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_C, h_C, matrix_bytes));

    int n_dim = N;
    void *kernel_args[] = { &d_A, &d_B, &d_C, &n_dim };

    const unsigned int tile_dim = 16;
    const unsigned int grid_dim = N / tile_dim; // 64 x 64 = 4096 blocks
    printf("[*] Dispatch Geometry: Grid=(%u, %u, 1) [%u blocks], Block=(%u, %u, 1) [256 threads/block]\n",
           grid_dim, grid_dim, grid_dim * grid_dim, tile_dim, tile_dim);
    printf("[*] Total Concurrent GPU Threads: %u threads\n", (grid_dim * grid_dim) * 256);

    // Warmup pass
    printf("[*] Running GPU warmup pass...\n");
    CHECK_CUDA(cuLaunchKernel(func,
                              grid_dim, grid_dim, 1,
                              tile_dim, tile_dim, 1,
                              0, 0,
                              kernel_args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    // Validar corrección del cálculo antes del benchmark
    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, matrix_bytes));
    printf("[*] Validating arithmetic correctness (Expected C[i][j] = 2048.000000)...\n");
    int mismatches = 0;
    for (size_t i = 0; i < num_elements; i++) {
        if (fabsf(h_C[i] - 2048.0f) > 1e-2f) {
            if (mismatches < 5) {
                printf("[-] MISMATCH at index %zu: GPU=%f, Expected=2048.000000\n", i, h_C[i]);
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        printf("[-] FAILED with %d mismatches! Aborting benchmark.\n", mismatches);
        return 1;
    }
    printf("[+] VERIFICATION PASSED: 1,048,576 / 1,048,576 floats mathematically perfect!\n");

    // Benchmark: 10 iteraciones
    const int ITERS = 10;
    printf("[*] Running %d timed benchmark iterations...\n", ITERS);

    double best_time = 1e9;
    double total_time = 0.0;

    for (int iter = 0; iter < ITERS; iter++) {
        double t0 = get_time_sec();
        CHECK_CUDA(cuLaunchKernel(func,
                                  grid_dim, grid_dim, 1,
                                  tile_dim, tile_dim, 1,
                                  0, 0,
                                  kernel_args, NULL));
        CHECK_CUDA(cuCtxSynchronize());
        double t1 = get_time_sec();
        double elapsed = t1 - t0;
        total_time += elapsed;
        if (elapsed < best_time) best_time = elapsed;
        double gflops = gflops_per_iter / elapsed;
        printf("    Iter %2d: %8.3f ms | %6.2f GFLOPS\n", iter + 1, elapsed * 1000.0, gflops);
    }

    double avg_time = total_time / ITERS;
    double avg_gflops = gflops_per_iter / avg_time;
    double peak_gflops = gflops_per_iter / best_time;

    printf("============================================================\n");
    printf("  SGEMM BENCHMARK RESULTS (Kepler GT 750M SM_30 Native)\n");
    printf("============================================================\n");
    printf("  Matrix Size:        %d x %d (FP32)\n", N, N);
    printf("  Operations / Pass:  2.147 GFLOP (2 * N^3)\n");
    printf("  Average Time:       %.3f ms\n", avg_time * 1000.0);
    printf("  Best Time:          %.3f ms\n", best_time * 1000.0);
    printf("  Average Throughput: %.2f GFLOPS\n", avg_gflops);
    printf("  Peak Throughput:    %.2f GFLOPS\n", peak_gflops);
    printf("  Theoretical Peak:   711.2 GFLOPS (GT 750M @ 926 MHz)\n");
    printf("  Efficiency:         %.1f%% of theoretical peak\n", (peak_gflops / 711.2) * 100.0);
    printf("============================================================\n");

    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cuFuncDestroy(func);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    return 0;
}
