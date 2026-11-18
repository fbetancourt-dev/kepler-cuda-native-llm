#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>

#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "[-] CUDA Error in %s: code %d\n", #call, err); \
        return 1; \
    } \
} while (0)

int main(void) {
    printf("============================================================\n");
    printf("  TEST PASO 2: LOCAL MEMORY & THREAD LOCAL STORAGE (TLS)\n");
    printf("  Kepler GT 750M SM_30 (Stack Frame 128B/thread, VRAM Spill)\n");
    printf("============================================================\n");

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    const char *cubin_path = "/home/fbetancourt/Gemini/Sandbox/cuda_kepler_native/spill_official.cubin";
    printf("[*] Loading CUBIN with stack spill from %s...\n", cubin_path);

    CUmodule mod;
    CHECK_CUDA(cuModuleLoad(&mod, cubin_path));

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_spill_kernel"));
    printf("[+] Kernel 'test_spill_kernel' loaded! TLS Stack allocated in VRAM!\n");

    const int NUM_BLOCKS = 32;
    const int THREADS_PER_BLOCK = 32;
    const int N = NUM_BLOCKS * THREADS_PER_BLOCK; // 1024 threads
    const size_t size = N * sizeof(float);

    CUdeviceptr d_in, d_out;
    CHECK_CUDA(cuMemAlloc(&d_in, size));
    CHECK_CUDA(cuMemAlloc(&d_out, size));

    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f; // All ones -> Expected sum = 32*1 + 992 = 1024.0f
        h_out[i] = 0.0f;
    }

    CHECK_CUDA(cuMemcpyHtoD(d_in, h_in, size));
    CHECK_CUDA(cuMemcpyHtoD(d_out, h_out, size));

    int n_val = N;
    void *kernel_args[] = { &d_in, &d_out, &n_val };

    printf("[*] Launching kernel with stack spills on GPU: Grid=%d blocks, Block=%d threads...\n",
           NUM_BLOCKS, THREADS_PER_BLOCK);

    CHECK_CUDA(cuLaunchKernel(func,
                              NUM_BLOCKS, 1, 1,
                              THREADS_PER_BLOCK, 1, 1,
                              0, 0,
                              kernel_args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_out, d_out, size));

    printf("[*] Validating 1024 floats with stack spill results (Expected: 1024.000000)...\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = 1024.0f;
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            if (errors < 5) printf("[-] MISMATCH at [%d]: GPU=%f, Expected=%f\n", i, h_out[i], expected);
            errors++;
        }
    }

    if (errors == 0) {
        printf("[+] SUCCESS! 1024/1024 floats verified perfectly!\n");
        printf("    Sample: out[0]=%f (Expected 1024.0), out[1023]=%f (Expected 1024.0)\n",
               h_out[0], h_out[1023]);
        printf("    Hardware Local Memory / TLS Stack Execution Verified on Silicon!\n");
    } else {
        printf("[-] FAILED with %d mismatches!\n", errors);
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    free(h_in);
    free(h_out);
    cuFuncDestroy(func);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    return errors;
}
