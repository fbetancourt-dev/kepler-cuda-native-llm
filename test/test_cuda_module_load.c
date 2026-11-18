#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
    printf("  KEPLER GK107 CUBIN FILE LOADER TEST (cuModuleLoad from Disk)\n");
    printf("  Binary: vector_add.cubin (256 bytes) on GeForce GT 750M\n");
    printf("============================================================\n");

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    const char *cubin_path = "/home/fbetancourt/Gemini/Sandbox/cuda_kepler_native/vector_add.cubin";
    printf("[*] Loading compiled Kepler binary from %s...\n", cubin_path);

    CUmodule mod;
    CHECK_CUDA(cuModuleLoad(&mod, cubin_path));
    printf("[+] cuModuleLoad() succeeded! Binary mapped into GPU VRAM.\n");

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "vector_add_kernel"));

    const int NUM_BLOCKS = 32;
    const int THREADS_PER_BLOCK = 32;
    const int N = NUM_BLOCKS * THREADS_PER_BLOCK; // 1024 elementos
    const size_t size = N * sizeof(uint32_t);

    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, size));
    CHECK_CUDA(cuMemAlloc(&d_B, size));
    CHECK_CUDA(cuMemAlloc(&d_C, size));

    uint32_t *h_A = (uint32_t *)malloc(size);
    uint32_t *h_B = (uint32_t *)malloc(size);
    uint32_t *h_C = (uint32_t *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i * 2 + 5;
        h_B[i] = 2000 - i;
        h_C[i] = 0xDEADBEEF;
    }

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size));
    CHECK_CUDA(cuMemcpyHtoD(d_C, h_C, size));

    void *kernel_args[] = { &d_A, &d_B, &d_C, NULL };

    printf("[*] Launching kernel loaded from .cubin (32 Blocks x 32 Threads = 1024 Threads)...\n");
    CHECK_CUDA(cuLaunchKernel(
        func,
        NUM_BLOCKS, 1, 1,
        THREADS_PER_BLOCK, 1, 1,
        0, 0,
        kernel_args,
        NULL
    ));

    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, size));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        uint32_t expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            if (errors < 5) {
                printf("    [FAIL] C[%d]: GPU=0x%08x != Expected=%u\n", i, h_C[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  >>> [SUCCESS] ALL 1024 ELEMENTS VERIFIED FROM STANDALONE .CUBIN FILE! <<<\n");
        printf("  Sample: C[0]=%u (Expected %u), C[1023]=%u (Expected %u)\n",
               h_C[0], h_A[0] + h_B[0], h_C[1023], h_A[1023] + h_B[1023]);
        printf("  Independent binary compilation and loading from disk is 100%% OPERATIONAL!\n");
    } else {
        printf("  [FAIL] Errors detected: %d / %d elements\n", errors, N);
    }
    printf("============================================================\n");

    CHECK_CUDA(cuModuleUnload(mod));
    CHECK_CUDA(cuMemFree(d_A));
    CHECK_CUDA(cuMemFree(d_B));
    CHECK_CUDA(cuMemFree(d_C));
    CHECK_CUDA(cuCtxDestroy(ctx));
    free(h_A);
    free(h_B);
    free(h_C);

    return (errors == 0) ? 0 : 1;
}
