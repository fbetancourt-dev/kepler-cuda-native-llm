#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>

#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "[-] CUDA Error en %s: código %d\n", #call, err); \
        return 1; \
    } \
} while (0)

int main(void) {
    printf("============================================================\n");
    printf("  NVIDIA OFFICIAL PTXAS ELF .CUBIN EXECUTION ON GT 750M (sm_30)\n");
    printf("  Direct Hardware Execution of NVIDIA ELF SASS via libcuda\n");
    printf("============================================================\n");

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    const char *cubin_path = "/home/fbetancourt/Gemini/Sandbox/cuda_kepler_native/vector_official.cubin";
    printf("[*] Loading NVIDIA official ELF .cubin from %s...\n", cubin_path);

    CUmodule mod;
    CHECK_CUDA(cuModuleLoad(&mod, cubin_path));
    printf("[+] cuModuleLoad() successfully parsed ELF 64-bit and mapped into VRAM!\n");

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "vec_add"));
    printf("[+] cuModuleGetFunction('vec_add') resolved entry symbol!\n");

    const int NUM_BLOCKS = 32;
    const int THREADS_PER_BLOCK = 32;
    const int N = NUM_BLOCKS * THREADS_PER_BLOCK; // 1024 elementos
    const size_t size = N * sizeof(float);

    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, size));
    CHECK_CUDA(cuMemAlloc(&d_B, size));
    CHECK_CUDA(cuMemAlloc(&d_C, size));

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 1.5f + 10.0f;
        h_B[i] = (float)i * 0.5f - 4.0f;
        h_C[i] = 0.0f;
    }

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size));
    CHECK_CUDA(cuMemcpyHtoD(d_C, h_C, size));

    uint64_t n_val = N;
    // Parámetros ABI: A, B, C, n
    void *kernel_args[] = { &d_A, &d_B, &d_C, &n_val, NULL };

    printf("[*] Launching kernel: Grid=(%d, 1, 1), Block=(%d, 1, 1), N=%d threads...\n",
           NUM_BLOCKS, THREADS_PER_BLOCK, N);

    CUevent start, stop;
    cuEventCreate(&start, 0);
    cuEventCreate(&stop, 0);

    cuEventRecord(start, 0);
    CHECK_CUDA(cuLaunchKernel(func,
                              NUM_BLOCKS, 1, 1,
                              THREADS_PER_BLOCK, 1, 1,
                              0, 0,
                              kernel_args, NULL));
    cuEventRecord(stop, 0);

    float elapsed_ms = 0.0f;
    cuEventElapsedTime(&elapsed_ms, start, stop);
    printf("[+] Kernel finished on GPU in %.3f ms!\n", elapsed_ms);

    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, size));

    printf("[*] Validating 1024 floating point results against CPU ground truth...\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-4f) {
            if (errors < 5) {
                printf("[-] MISMATCH at [%d]: GPU=%f, Expected=%f\n", i, h_C[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("[+] SUCCESS! 1024/1024 floats verified perfectly! NVIDIA PTXAS .cubin executed flawlessly on GT 750M!\n");
        printf("    Sample: C[0]=%f (Expected %f), C[1023]=%f (Expected %f)\n",
               h_C[0], h_A[0] + h_B[0], h_C[1023], h_A[1023] + h_B[1023]);
    } else {
        printf("[-] FAILED with %d mismatches!\n", errors);
    }

    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    return errors == 0 ? 0 : 1;
}
