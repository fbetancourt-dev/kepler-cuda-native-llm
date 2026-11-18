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

#define KEPLER_NOP_LO 0x00001de4
#define KEPLER_NOP_HI 0x40000000

/*
 * Kepler GK107 (sm_30) Hardware Vector Addition Kernel:
 *   C[gid] = A[gid] + B[gid]
 *
 * ABI Constant Buffer 0:
 *   c0[0x140] / c0[0x144] : uint64_t d_A
 *   c0[0x148] / c0[0x14c] : uint64_t d_B
 *   c0[0x150] / c0[0x154] : uint64_t d_C
 */
static const uint32_t kepler_vec_add_kernel[256] = {
    /* === BUNDLE 0 (Bytes 0x00..0x3F): Thread Coordinates + Pointers === */
    // Control Word: 7 instrucciones
    0x00000007 | (0x24 << 4) | (0x24 << 12) | (0x24 << 20) | ((uint32_t)0x2eULL << 28),
    0x20000000 | (0x2e >> 4),
    // [1] S2R R0, SR_TID.X
    0x84001c04, 0x2c000000,
    // [2] S2R R1, SR_CTAID.X
    0x94005c04, 0x2c000000,
    // [3] SHL R2, R1, 5  (R2 = blockIdx.x * 32)
    0x14109c03, 0x6000c000,
    // [4] ADD R0, R0, R2 (R0 = gid = threadIdx.x + blockIdx.x * 32)
    0x08001c03, 0x48000000,
    // [5] SHL R3, R0, 2  (R3 = gid * 4 bytes)
    0x0800dc03, 0x6000c000,
    // [6] MOV R4, c[0x0][0x140] (d_A low)
    0x00011de4, 0x28004005,
    // [7] MOV R5, c[0x0][0x144] (d_A high)
    0x10015de4, 0x28004005,

    /* === BUNDLE 1 (Bytes 0x40..0x7F): Pointers B and C + Addresses === */
    // Control Word: 7 instrucciones
    0x00000007 | (0x24 << 4) | (0x24 << 12) | (0x24 << 20) | ((uint32_t)0x2eULL << 28),
    0x20000000 | (0x2e >> 4),
    // [1] MOV R6, c[0x0][0x148] (d_B low)
    0x20019de4, 0x28004005,
    // [2] MOV R7, c[0x0][0x14c] (d_B high)
    0x3001dde4, 0x28004005,
    // [3] MOV R8, c[0x0][0x150] (d_C low)
    0x40021de4, 0x28004005,
    // [4] MOV R9, c[0x0][0x154] (d_C high)
    0x50025de4, 0x28004005,
    // [5] ADD R4, R4, R3 (R4 = &A[gid])
    0x0c411c03, 0x48000000,
    // [6] ADD R6, R6, R3 (R6 = &B[gid])
    0x0c619c03, 0x48000000,
    // [7] ADD R8, R8, R3 (R8 = &C[gid])
    0x0c821c03, 0x48000000,

    /* === BUNDLE 2 (Bytes 0x80..0xBF): Global Loads, Compute, Store, EXIT === */
    // Control Word: 7 instrucciones (LD, LD, stall, ADD, ST, EXIT, NOP)
    0x00000007 | (0x2e << 4) | (0x2e << 12) | (0x2e << 20) | ((uint32_t)0x2eULL << 28),
    0x20000000 | (0x2e >> 4),
    // [1] LD.E R10, [R4] (Load A[gid])
    0x00429c85, 0x84000000,
    // [2] LD.E R11, [R6] (Load B[gid])
    0x0062dc85, 0x84000000,
    // [3] ADD R12, R10, R11 (R12 = A[gid] + B[gid])
    0x2ca31c03, 0x48000000,
    // [4] ST.E [R8], R12   (C[gid] = R12)
    0x00831c85, 0x94000000,
    // [5] EXIT
    0x00001de7, 0x80000000,
    // [6] Canonical Kepler NOP
    KEPLER_NOP_LO, KEPLER_NOP_HI,
    // [7] Canonical Kepler NOP
    KEPLER_NOP_LO, KEPLER_NOP_HI
};

static int run_vector_add_test(CUfunction func, int num_blocks, int threads_per_block) {
    int N = num_blocks * threads_per_block;
    size_t size = N * sizeof(uint32_t);

    printf("\n>>> Running Vector Add: C[i] = A[i] + B[i] for N = %d elements (%d blocks x %d threads)...\n",
           N, num_blocks, threads_per_block);

    // Alojar buffers de GPU
    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, size));
    CHECK_CUDA(cuMemAlloc(&d_B, size));
    CHECK_CUDA(cuMemAlloc(&d_C, size));

    // Inicializar datos en Host
    uint32_t *h_A = (uint32_t *)malloc(size);
    uint32_t *h_B = (uint32_t *)malloc(size);
    uint32_t *h_C = (uint32_t *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i * 3 + 1;
        h_B[i] = 1000 - i * 2;
        h_C[i] = 0xDEADBEEF; // Centinela
    }

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size));
    CHECK_CUDA(cuMemcpyHtoD(d_C, h_C, size));

    // Parámetros del kernel: d_A, d_B, d_C
    void *kernel_args[] = { &d_A, &d_B, &d_C, NULL };

    CHECK_CUDA(cuLaunchKernel(
        func,
        num_blocks, 1, 1,
        threads_per_block, 1, 1,
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
                printf("    [FAIL] C[%d]: GPU=0x%08x (%u) != Expected=%u (A=%u, B=%u)\n",
                       i, h_C[i], h_C[i], expected, h_A[i], h_B[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  [PASS] All %d elements computed EXACTLY: C[i] == A[i] + B[i]! (Sample: C[0]=%u, C[%d]=%u)\n",
               N, h_C[0], N-1, h_C[N-1]);
    } else {
        printf("  [FAIL] %d / %d elements with mismatch!\n", errors, N);
    }

    CHECK_CUDA(cuMemFree(d_A));
    CHECK_CUDA(cuMemFree(d_B));
    CHECK_CUDA(cuMemFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return errors;
}

int main(void) {
    printf("============================================================\n");
    printf("  KEPLER GK107 NATIVE VECTOR ADDITION COMPUTE TEST\n");
    printf("  Formula: C[i] = A[i] + B[i] (Concurrent Global VRAM LD.E/ST.E)\n");
    printf("  Hardware: NVIDIA GeForce GT 750M (sm_30, 384 CUDA Cores)\n");
    printf("============================================================\n");

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK_CUDA(cuModuleLoadData(&mod, kepler_vec_add_kernel));
    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "vector_add_kernel"));

    int total_errors = 0;

    // 1 Warp (32 elementos)
    total_errors += run_vector_add_test(func, 1, 32);

    // 4 Bloques (128 elementos)
    total_errors += run_vector_add_test(func, 4, 32);

    // 8 Bloques (256 elementos)
    total_errors += run_vector_add_test(func, 8, 32);

    // 16 Bloques (512 elementos)
    total_errors += run_vector_add_test(func, 16, 32);

    // 32 Bloques (1024 elementos)
    total_errors += run_vector_add_test(func, 32, 32);

    printf("\n============================================================\n");
    if (total_errors == 0) {
        printf("  >>> [SUCCESS] 100%% HARDWARE VECTOR ADD VERIFIED ON KEPLER! <<<\n");
        printf("  Multi-buffer DMA + SASS LD.E + Integer ALU + ST.E fully working!\n");
    } else {
        printf("  [FAIL] %d errors detected.\n", total_errors);
    }
    printf("============================================================\n");

    CHECK_CUDA(cuModuleUnload(mod));
    CHECK_CUDA(cuCtxDestroy(ctx));
    return (total_errors == 0) ? 0 : 1;
}
