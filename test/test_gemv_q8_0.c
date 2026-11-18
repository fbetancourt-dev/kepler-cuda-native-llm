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

// ====================================================================
// CUANTIZADOR DE REFERENCIA EN CPU
// ====================================================================

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
// REFERENCIA CPU (PRECISIÓN DOUBLE)
// ====================================================================

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

// ====================================================================
// TEST & BENCHMARK EN SILICIO REAL
// ====================================================================

int main(void) {
    printf("====================================================================\n");
    printf("  GEMV Q8_0 (W8A32) KERNEL SILICON BENCHMARK & ACCURACY SUITE\n");
    printf("  Memory Compression: 3.55x (36 bytes / 32 weights vs 128 bytes FP32)\n");
    printf("  Silicon: NVIDIA GeForce GT 750M (Kepler GK107 sm_30 Native)\n");
    printf("====================================================================\n");

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    char dev_name[256];
    CHECK_CUDA(cuDeviceGetName(dev_name, sizeof(dev_name), dev));
    printf("[+] Target Hardware: %s\n", dev_name);

    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK_CUDA(cuModuleLoad(&mod, "gemv_q8_0_official.cubin"));
    CUfunction fn_gemv_q8_0;
    CHECK_CUDA(cuModuleGetFunction(&fn_gemv_q8_0, mod, "gemv_q8_0_kernel"));
    printf("[+] CUBIN 'gemv_q8_0_official.cubin' loaded and function resolved.\n");

    // Dimensiones representativas de TinyLlama-1.1B
    const int M = 2048;  // Fila de Wq o Wo
    const int K = 2048;  // Dimensión oculta
    const int num_blocks = K / 32; // 64 bloques

    size_t fp32_weight_bytes = (size_t)M * K * sizeof(float);           // 16.0 MB
    size_t q8_weight_bytes   = (size_t)M * num_blocks * sizeof(block_q8_0); // 4.72 MB (36 B * 64 * 2048 = 4,718,592 B)
    size_t act_bytes         = (size_t)K * sizeof(float);               // 8 KB
    size_t out_bytes         = (size_t)M * sizeof(float);               // 8 KB

    printf("[*] Matrix: %d x %d | FP32: %.2f MB -> Q8_0: %.2f MB (Compression Ratio: %.2fx)\n",
           M, K, (double)fp32_weight_bytes / (1024.0 * 1024.0),
           (double)q8_weight_bytes / (1024.0 * 1024.0),
           (double)fp32_weight_bytes / (double)q8_weight_bytes);

    // Alojar memoria host
    float *h_A_fp32 = (float *)malloc(fp32_weight_bytes);
    block_q8_0 *h_A_q8 = (block_q8_0 *)malloc(q8_weight_bytes);
    float *h_x = (float *)malloc(act_bytes);
    float *h_y_cpu = (float *)malloc(out_bytes);
    float *h_y_gpu = (float *)malloc(out_bytes);

    // Inicializar pesos sintéticos y vector de activación
    srand(2026);
    for (size_t i = 0; i < (size_t)M * K; i++) {
        h_A_fp32[i] = (((float)(rand() % 2000) - 1000.0f) * 0.001f);
    }
    for (int i = 0; i < K; i++) {
        h_x[i] = sinf((float)i * 0.05f) * 0.5f;
    }

    // Cuantizar pesos a Q8_0
    printf("[*] Quantizing weights on host to Q8_0 blocks...\n");
    for (int r = 0; r < M; r++) {
        quantize_row_q8_0(h_A_q8 + (size_t)r * num_blocks, h_A_fp32 + (size_t)r * K, K);
    }

    // Calcular referencia en CPU (double precision)
    printf("[*] Computing CPU Ground Truth for Q8_0 GEMV (double precision)...\n");
    cpu_gemv_q8_0(h_y_cpu, h_A_q8, h_x, M, K);

    // Alojar tensores en GPU VRAM
    CUdeviceptr d_A_q8, d_x, d_y;
    CHECK_CUDA(cuMemAlloc(&d_A_q8, q8_weight_bytes));
    CHECK_CUDA(cuMemAlloc(&d_x, act_bytes));
    CHECK_CUDA(cuMemAlloc(&d_y, out_bytes));

    CHECK_CUDA(cuMemcpyHtoD(d_A_q8, h_A_q8, q8_weight_bytes));
    CHECK_CUDA(cuMemcpyHtoD(d_x, h_x, act_bytes));

    // Configuración de Grid: 1 warp por fila (32 hilos en X, 4 warps por bloque en Y)
    int threads_x = 32;
    int warps_per_block = 4;
    int blocks_x = (M + warps_per_block - 1) / warps_per_block;
    int m_arg = M;
    int num_blocks_arg = num_blocks;

    void *args[] = { &d_y, &d_A_q8, &d_x, &m_arg, &num_blocks_arg };

    printf("[*] Launching GEMV Q8_0 on GPU: Grid=%d blocks, Block=(%d, %d)...\n",
           blocks_x, threads_x, warps_per_block);

    CHECK_CUDA(cuLaunchKernel(fn_gemv_q8_0, blocks_x, 1, 1, threads_x, warps_per_block, 1, 0, 0, args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_y_gpu, d_y, out_bytes));

    // Validación numérica rigurosa
    int nan_count = 0;
    int inf_count = 0;
    float max_abs_diff = 0.0f;
    for (int i = 0; i < M; i++) {
        if (isnan(h_y_gpu[i])) nan_count++;
        if (isinf(h_y_gpu[i])) inf_count++;
        float diff = fabsf(h_y_gpu[i] - h_y_cpu[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    printf("[*] Accuracy Assessment:\n");
    printf("    Non-finite Values: %d (NaNs: %d, Infs: %d)\n", nan_count + inf_count, nan_count, inf_count);
    printf("    Max Absolute Diff vs CPU Q8_0: %e\n", max_abs_diff);

    if (nan_count > 0 || inf_count > 0 || max_abs_diff > 1e-4f) {
        fprintf(stderr, "[-] GEMV Q8_0 ACCURACY VERIFICATION FAILED!\n");
        return 1;
    }
    printf("[+] NUMERIC VERIFICATION PASSED! GPU matches CPU within FP32 machine precision.\n");

    // Benchmark de latencia (50 iteraciones amortizadas)
    const int iterations = 50;
    printf("[*] Running benchmark (%d iterations with amortized GPU sync)...\n", iterations);

    double t_start = get_time_sec();
    for (int it = 0; it < iterations; it++) {
        CHECK_CUDA(cuLaunchKernel(fn_gemv_q8_0, blocks_x, 1, 1, threads_x, warps_per_block, 1, 0, 0, args, NULL));
    }
    CHECK_CUDA(cuCtxSynchronize());
    double t_end = get_time_sec();

    double total_time_ms = (t_end - t_start) * 1000.0;
    double avg_time_ms = total_time_ms / (double)iterations;
    double bytes_per_gemv = (double)q8_weight_bytes + act_bytes + out_bytes;
    double effective_bw_gbps = (bytes_per_gemv / (avg_time_ms * 1e-3)) / (1e9);

    printf("--------------------------------------------------------------------\n");
    printf("  PERFORMANCE METRICS (GEMV Q8_0 MATRIX %d x %d):\n", M, K);
    printf("  Total Time (%d runs) : %.3f ms\n", iterations, total_time_ms);
    printf("  Average Latency      : %.3f ms\n", avg_time_ms);
    printf("  Throughput           : %.1f GEMV operations/sec\n", 1000.0 / avg_time_ms);
    printf("  Effective Bandwidth  : %.2f GB/s\n", effective_bw_gbps);
    printf("--------------------------------------------------------------------\n");

    // ====================================================================
    // CASO 2: MATRIZ MLP FFN (M = 5632, K = 2048) - Matriz más pesada
    // ====================================================================
    const int M2 = 5632;
    size_t q8_w_bytes2 = (size_t)M2 * num_blocks * sizeof(block_q8_0); // ~12.97 MB (vs 46.1 MB en FP32)
    size_t out_bytes2 = (size_t)M2 * sizeof(float);

    printf("\n[*] ------------------------------------------------------------\n");
    printf("[*] CASO 2: Matriz MLP FFN (%d x %d)\n", M2, K);
    printf("[*] FP32: %.2f MB -> Q8_0: %.2f MB (Compression Ratio: 3.56x)\n",
           (double)(M2 * K * sizeof(float)) / (1024.0 * 1024.0),
           (double)q8_w_bytes2 / (1024.0 * 1024.0));

    block_q8_0 *h_A_q8_2 = (block_q8_0 *)malloc(q8_w_bytes2);
    float *h_y_cpu2 = (float *)malloc(out_bytes2);
    float *h_y_gpu2 = (float *)malloc(out_bytes2);
    float *temp_row = (float *)malloc(K * sizeof(float));

    for (int r = 0; r < M2; r++) {
        for (int c = 0; c < K; c++) {
            temp_row[c] = (((float)((r + c) % 2000) - 1000.0f) * 0.001f);
        }
        quantize_row_q8_0(h_A_q8_2 + (size_t)r * num_blocks, temp_row, K);
    }
    free(temp_row);

    cpu_gemv_q8_0(h_y_cpu2, h_A_q8_2, h_x, M2, K);

    CUdeviceptr d_A_q8_2, d_y2;
    CHECK_CUDA(cuMemAlloc(&d_A_q8_2, q8_w_bytes2));
    CHECK_CUDA(cuMemAlloc(&d_y2, out_bytes2));
    CHECK_CUDA(cuMemcpyHtoD(d_A_q8_2, h_A_q8_2, q8_w_bytes2));

    int blocks_x2 = (M2 + warps_per_block - 1) / warps_per_block;
    int m_arg2 = M2;
    void *args2[] = { &d_y2, &d_A_q8_2, &d_x, &m_arg2, &num_blocks_arg };

    CHECK_CUDA(cuLaunchKernel(fn_gemv_q8_0, blocks_x2, 1, 1, threads_x, warps_per_block, 1, 0, 0, args2, NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_y_gpu2, d_y2, out_bytes2));

    int nan2 = 0, inf2 = 0;
    float max_diff2 = 0.0f;
    for (int i = 0; i < M2; i++) {
        if (isnan(h_y_gpu2[i])) nan2++;
        if (isinf(h_y_gpu2[i])) inf2++;
        float d = fabsf(h_y_gpu2[i] - h_y_cpu2[i]);
        if (d > max_diff2) max_diff2 = d;
    }
    printf("    [Result] NaNs: %d, Infs: %d | Max Abs Diff: %e\n", nan2, inf2, max_diff2);
    if (nan2 > 0 || inf2 > 0 || max_diff2 > 1e-4f) {
        fprintf(stderr, "[-] CASO 2 FAILED!\n");
        return 1;
    }
    printf("[+] CASO 2 PASSED 100%% IN SILICON!\n");

    // Benchmark Caso 2
    t_start = get_time_sec();
    for (int it = 0; it < iterations; it++) {
        CHECK_CUDA(cuLaunchKernel(fn_gemv_q8_0, blocks_x2, 1, 1, threads_x, warps_per_block, 1, 0, 0, args2, NULL));
    }
    CHECK_CUDA(cuCtxSynchronize());
    t_end = get_time_sec();
    total_time_ms = (t_end - t_start) * 1000.0;
    avg_time_ms = total_time_ms / (double)iterations;
    double bw2 = (((double)q8_w_bytes2 + act_bytes + out_bytes2) / (avg_time_ms * 1e-3)) / 1e9;

    printf("--------------------------------------------------------------------\n");
    printf("  PERFORMANCE METRICS (MLP FFN Q8_0 MATRIX %d x %d):\n", M2, K);
    printf("  Average Latency      : %.3f ms (vs ~9.5 ms en FP32)\n", avg_time_ms);
    printf("  Throughput           : %.1f GEMVs/sec\n", 1000.0 / avg_time_ms);
    printf("  Effective Bandwidth  : %.2f GB/s\n", bw2);
    printf("--------------------------------------------------------------------\n");

    cuMemFree(d_A_q8_2);
    cuMemFree(d_y2);
    free(h_A_q8_2);
    free(h_y_cpu2);
    free(h_y_gpu2);

    cuMemFree(d_A_q8);
    cuMemFree(d_x);
    cuMemFree(d_y);
    free(h_A_fp32);
    free(h_A_q8);
    free(h_x);
    free(h_y_cpu);
    free(h_y_gpu);

    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    printf("\n[+] ALL GEMV Q8_0 TESTS PASSED 100%% ON PHYSICAL SILICON!\n");
    return 0;
}
