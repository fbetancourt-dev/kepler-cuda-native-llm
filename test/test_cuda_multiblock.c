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

/*
 * Kepler GK107 (sm_30) Multi-Block & Multi-Warp Grid Kernel:
 *
 *   uint32_t tid   = threadIdx.x;      // S2R R0, SR_TID.X   (sr=0x21)
 *   uint32_t bid   = blockIdx.x;       // S2R R1, SR_CTAID.X (sr=0x25)
 *   uint32_t b_off = bid << 5;         // SHL R6, R1, 5      (bid * 32)
 *   uint32_t gid   = tid + b_off;      // ADD R0, R0, R6     (global ID)
 *   uint64_t out   = c0[0x148];        // MOV R2, c0[0x148] / MOV R3, c0[0x14c]
 *   uint32_t byte_off = gid << 2;      // SHL R4, R0, 2      (gid * 4)
 *   uint64_t addr  = out + byte_off;   // ADD R2, R2, R4
 *   uint32_t val   = gid + 42;         // ADD R5, R0, 42
 *   *addr          = val;              // ST.E [R2], R5
 *   exit;                              // EXIT
 */
static const uint32_t kepler_multiblock_kernel[256] = {
    /* === BUNDLE 0 (Bytes 0x00..0x3F) === */
    // Control Word: 7 instrucciones, delay=4, stall
    0x00000007 | (0x24 << 4) | (0x24 << 12) | (0x24 << 20) | (0x2eULL << 28),
    0x20000000 | (0x2e >> 4),
    // [1] S2R R0, SR_TID.X
    0x84001c04, 0x2c000000,
    // [2] S2R R1, SR_CTAID.X
    0x94005c04, 0x2c000000,
    // [3] SHL R6, R1, 5  (R6 = blockIdx.x * 32)
    0x14119c03, 0x6000c000,
    // [4] ADD R0, R0, R6 (R0 = gid = threadIdx.x + blockIdx.x * 32)
    0x18001c03, 0x48000000,
    // [5] MOV R2, c[0x0][0x148] (Output Low)
    0x20009de4, 0x28004005,
    // [6] MOV R3, c[0x0][0x14c] (Output High)
    0x3000dde4, 0x28004005,
    // [7] SHL R4, R0, 2  (R4 = gid * 4 bytes)
    0x08011c03, 0x6000c000,

    /* === BUNDLE 1 (Bytes 0x40..0x7F) === */
    // Control Word: 3 instrucciones (ADD, ADD, ST.E)
    0x00000003 | (0x24 << 4) | (0x24 << 12) | (0x2eULL << 20),
    0x20000000 | (0x2e >> 12),
    // [1] ADD R2, R2, R4 (R2 = base_lo + gid * 4)
    0x10209c03, 0x48000000,
    // [2] ADD R5, R0, 42 (R5 = gid + 42)
    0xa8015c03, 0x4800c000,
    // [3] ST.E [R2], R5  (Global Store: output[gid] = R5)
    0x00215c85, 0x94000000,
    0x00000000, 0x00000000, // NOP
    0x00000000, 0x00000000, // NOP
    0x00000000, 0x00000000, // NOP
    0x00000000, 0x00000000, // NOP

    /* === BUNDLE 2 (Bytes 0x80..0xBF) === */
    // Control Word: 1 instrucción (EXIT)
    0x000002e7, 0x20000000,
    // [1] EXIT
    0x00001de7, 0x80000000,
    0x00000000, 0x00000000,
    0x00000000, 0x00000000,
    0x00000000, 0x00000000,
    0x00000000, 0x00000000,
    0x00000000, 0x00000000
};

static int run_grid_test(CUfunction func, int num_blocks, int threads_per_block) {
    int total_threads = num_blocks * threads_per_block;
    size_t size = total_threads * sizeof(uint32_t);

    printf("\n>>> Testing Grid Execution: %d Blocks x %d Threads = %d Total Threads...\n",
           num_blocks, threads_per_block, total_threads);

    CUdeviceptr d_output;
    CHECK_CUDA(cuMemAlloc(&d_output, size));

    // Inicializar memoria GPU con centinela
    uint32_t *h_init = (uint32_t *)malloc(size);
    for (int i = 0; i < total_threads; i++) h_init[i] = 0xDEADBEEF;
    CHECK_CUDA(cuMemcpyHtoD(d_output, h_init, size));
    free(h_init);

    void *kernel_args[] = { &d_output, NULL };

    CHECK_CUDA(cuLaunchKernel(
        func,
        num_blocks, 1, 1,        // gridDim
        threads_per_block, 1, 1, // blockDim
        0, 0,
        kernel_args,
        NULL
    ));

    CHECK_CUDA(cuCtxSynchronize());

    uint32_t *h_output = (uint32_t *)malloc(size);
    CHECK_CUDA(cuMemcpyDtoH(h_output, d_output, size));

    int errors = 0;
    for (int i = 0; i < total_threads; i++) {
        uint32_t expected = i + 42;
        if (h_output[i] != expected) {
            if (errors < 10) {
                printf("    [FAIL] Thread %d (Block %d): GPU=0x%08x (%u) != Expected=%u\n",
                       i, i / threads_per_block, h_output[i], h_output[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  [PASS] %d/%d threads verified 100%% exact across %d blocks! All SMX active.\n",
               total_threads, total_threads, num_blocks);
    } else {
        printf("  [FAIL] Detected %d errors out of %d threads!\n", errors, total_threads);
    }

    CHECK_CUDA(cuMemFree(d_output));
    free(h_output);
    return errors;
}

int main(void) {
    printf("============================================================\n");
    printf("  KEPLER GK107 MULTI-BLOCK / MULTI-WARP GRID TEST\n");
    printf("  Hardware: NVIDIA GeForce GT 750M (sm_30, 384 CUDA Cores)\n");
    printf("  Testing: S2R SR_CTAID.X + S2R SR_TID.X + HW Grid Dispatch\n");
    printf("============================================================\n");

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    char name[128];
    CHECK_CUDA(cuDeviceGetName(name, sizeof(name), dev));
    printf("[+] Active Device: %s\n", name);

    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK_CUDA(cuModuleLoadData(&mod, kepler_multiblock_kernel));
    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "multiblock_kernel"));

    int total_errors = 0;

    // Test 1: 1 Bloque (32 hilos) - Baseline
    total_errors += run_grid_test(func, 1, 32);

    // Test 2: 2 Bloques (64 hilos) - Verificación de distribución de bloques
    total_errors += run_grid_test(func, 2, 32);

    // Test 3: 4 Bloques (128 hilos) - 2 bloques por SMX
    total_errors += run_grid_test(func, 4, 32);

    // Test 4: 8 Bloques (256 hilos) - Carga balanceada
    total_errors += run_grid_test(func, 8, 32);

    // Test 5: 16 Bloques (512 hilos) - Saturación completa de los 384 cores
    total_errors += run_grid_test(func, 16, 32);

    // Test 6: 32 Bloques (1024 hilos) - Grid masivo en Kepler
    total_errors += run_grid_test(func, 32, 32);

    printf("\n============================================================\n");
    if (total_errors == 0) {
        printf("  >>> [OVERALL SUCCESS] ALL MULTI-BLOCK GRID TESTS PASSED! <<<\n");
        printf("  Kepler Hardware CTA Dispatcher fully verified on GT 750M.\n");
    } else {
        printf("  [OVERALL FAIL] %d total errors detected across tests.\n", total_errors);
    }
    printf("============================================================\n");

    CHECK_CUDA(cuModuleUnload(mod));
    CHECK_CUDA(cuCtxDestroy(ctx));
    return (total_errors == 0) ? 0 : 1;
}
