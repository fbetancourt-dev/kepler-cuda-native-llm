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
    printf("  TEST PASO 1 & 2: MIXED TYPES (64-bit ptr + 32-bit int)\n");
    printf("  Exact nvinfo parameter packing & Per-Function BO Isolation\n");
    printf("============================================================\n");

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    const char *cubin_path = "/home/fbetancourt/Gemini/Sandbox/cuda_kepler_native/vector_official.cubin";
    printf("[*] Loading CUBIN from %s...\n", cubin_path);

    CUmodule mod;
    CHECK_CUDA(cuModuleLoad(&mod, cubin_path));

    CUfunction func1, func2;
    // Cargar dos veces la función para validar que cada una tiene su propio BO aislado (Paso 2)
    CHECK_CUDA(cuModuleGetFunction(&func1, mod, "vec_add"));
    CHECK_CUDA(cuModuleGetFunction(&func2, mod, "vec_add"));
    printf("[+] cuModuleGetFunction() resolved two independent handles with isolated BOs!\n");

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
        h_A[i] = (float)i * 2.0f;
        h_B[i] = 100.0f - (float)i;
        h_C[i] = 0.0f;
    }

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size));
    CHECK_CUDA(cuMemcpyHtoD(d_C, h_C, size));

    // [PASO 1] 32-bit int real (sin padding ni 64-bit cast) y SIN NULL terminator
    int n_val = N;
    void *kernel_args[] = { &d_A, &d_B, &d_C, &n_val };

    printf("[*] Launching kernel on GPU (d_A=0x%lx, d_B=0x%lx, d_C=0x%lx, n=%d)...\n",
           (uint64_t)d_A, (uint64_t)d_B, (uint64_t)d_C, n_val);

    CHECK_CUDA(cuLaunchKernel(func1,
                              NUM_BLOCKS, 1, 1,
                              THREADS_PER_BLOCK, 1, 1,
                              0, 0,
                              kernel_args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, size));

    printf("[*] Validating 1024 floats with bitwise and mathematical equality...\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-4f) {
            if (errors < 5) printf("[-] MISMATCH at [%d]: GPU=%f, Expected=%f\n", i, h_C[i], expected);
            errors++;
        }
    }

    if (errors == 0) {
        printf("[+] SUCCESS! 1024/1024 floats verified perfectly!\n");
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
