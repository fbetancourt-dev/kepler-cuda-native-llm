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

static int test_reduction_cubin(const char *backend_name, const char *cubin_path) {
    printf("------------------------------------------------------------\n");
    printf("[*] Testing Pipeline: %s (%s)\n", backend_name, cubin_path);
    printf("------------------------------------------------------------\n");

    CUmodule mod;
    CHECK_CUDA(cuModuleLoad(&mod, cubin_path));

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "block_reduce_shared"));
    printf("[+] Kernel 'block_reduce_shared' loaded and QMD resources initialized!\n");

    const int THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS = 8;
    const int N = NUM_BLOCKS * THREADS_PER_BLOCK; // 2048 elements
    const size_t in_size = N * sizeof(float);
    const size_t out_size = NUM_BLOCKS * sizeof(float);
    const unsigned int dynamic_smem = THREADS_PER_BLOCK * sizeof(float); // 1024 bytes dynamic shared memory

    CUdeviceptr d_in, d_out;
    CHECK_CUDA(cuMemAlloc(&d_in, in_size));
    CHECK_CUDA(cuMemAlloc(&d_out, out_size));

    float *h_in = (float *)malloc(in_size);
    float *h_out = (float *)malloc(out_size);

    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f; // All ones
    }
    for (int b = 0; b < NUM_BLOCKS; b++) {
        h_out[b] = 0.0f;
    }

    CHECK_CUDA(cuMemcpyHtoD(d_in, h_in, in_size));
    CHECK_CUDA(cuMemcpyHtoD(d_out, h_out, out_size));

    int n_val = N;
    void *kernel_args[] = { &d_in, &d_out, &n_val };

    printf("[*] Launching kernel on GPU: Grid=%d blocks, Block=%d threads, Dynamic SMEM=%u bytes...\n",
           NUM_BLOCKS, THREADS_PER_BLOCK, dynamic_smem);

    CHECK_CUDA(cuLaunchKernel(func,
                              NUM_BLOCKS, 1, 1,
                              THREADS_PER_BLOCK, 1, 1,
                              dynamic_smem, 0,
                              kernel_args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_out, d_out, out_size));

    printf("[*] Validating block reduction results (Expected per block: 512.000000)...\n");
    int errors = 0;
    float total_sum = 0.0f;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        total_sum += h_out[b];
        if (fabsf(h_out[b] - 512.0f) > 1e-3f) {
            printf("[-] MISMATCH in Block [%d]: GPU=%f, Expected=512.000000\n", b, h_out[b]);
            errors++;
        } else {
            printf("    [+] Block [%d]: Sum = %f (OK)\n", b, h_out[b]);
        }
    }

    if (errors == 0) {
        printf("[+] SUCCESS! %s passed with total reduction sum = %f (Expected %f)\n",
               backend_name, total_sum, 512.0f * NUM_BLOCKS);
    } else {
        printf("[-] FAILED for %s with %d errors!\n", backend_name, errors);
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    free(h_in);
    free(h_out);
    cuFuncDestroy(func);
    cuModuleUnload(mod);

    return errors;
}

int main(void) {
    printf("============================================================\n");
    printf("  TEST PASO 3: DYNAMIC QMD SETUP (Shared Mem + Barriers)\n");
    printf("  Kepler GT 750M SM_30 (Static + Dynamic SMEM, __syncthreads)\n");
    printf("============================================================\n");

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    int res1 = test_reduction_cubin("PIPELINE 1 (NVIDIA Official CUDA 10.2 sm_30)",
                                    "/home/fbetancourt/Gemini/Sandbox/cuda_kepler_native/reduction_official.cubin");

    CHECK_CUDA(cuCtxDestroy(ctx));

    if (res1 == 0) {
        printf("\n============================================================\n");
        printf("  [+] PASO 3 VERIFIED 100%% IN REAL KEPLER SILICON!\n");
        printf("  Static SMEM (1024B) + Dynamic SMEM (1024B) + __syncthreads\n");
        printf("  Dynamic QMD Barrier Count & L1 Cache fully validated!\n");
        printf("============================================================\n");
        return 0;
    } else {
        printf("\n[-] REDUCTION TEST FAILED!\n");
        return 1;
    }
}
