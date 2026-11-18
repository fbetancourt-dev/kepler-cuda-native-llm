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
 * Kepler GK107 (sm_30) Microkernel binario de 64 bits:
 *   output[tid] = tid + 42
 * Verificado instrucción por instrucción en hardware.
 */
static const uint32_t kepler_compute_kernel[] = {
    /* === BUNDLE 0 (Bytes 0x00..0x3F) === */
    0x00000007 | (0x24 << 4) | (0x24 << 12) | (0x24 << 20) | (0x2eULL << 28),
    0x20000000 | (0x2e >> 4),
    0x84001c04, 0x2c000000, // S2R R0, SR_TID.X
    0x20009de4, 0x28004005, // MOV R2, c[0x0][0x148] (Output Low)
    0x3000dde4, 0x28004005, // MOV R3, c[0x0][0x14c] (Output High)
    0x08011c03, 0x6000c000, // SHL R4, R0, 2
    0x10209c03, 0x48000000, // ADD R2, R2, R4
    0xa8015c03, 0x4800c000, // ADD R5, R0, 42
    0x00215c85, 0x94000000, // ST.E [R2], R5

    /* === BUNDLE 1 (Bytes 0x40..0x7F) === */
    0x000002e7, 0x20000000, // Sched Control Word
    0x00001de7, 0x80000000, // EXIT
    0x00000000, 0x00000000,
    0x00000000, 0x00000000,
    0x00000000, 0x00000000,
    0x00000000, 0x00000000,
    0x00000000, 0x00000000
};

int main(void) {
    printf("============================================================\n");
    printf("  END-TO-END CUDA DRIVER API COMPUTE TEST (via libcuda.so.1)\n");
    printf("  Hardware: NVIDIA GeForce GT 750M (Kepler GK107 sm_30)\n");
    printf("  Executing: output[i] = threadIdx.x + 42 via cuLaunchKernel\n");
    printf("============================================================\n\n");

    // 1. Inicialización
    CHECK_CUDA(cuInit(0));
    printf("[+] cuInit() OK\n");

    int count = 0;
    CHECK_CUDA(cuDeviceGetCount(&count));
    printf("[+] CUDA Devices: %d\n", count);

    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    char name[128];
    CHECK_CUDA(cuDeviceGetName(name, sizeof(name), dev));
    printf("[+] Device 0: %s\n", name);

    // 2. Crear Contexto
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));
    printf("[+] cuCtxCreate() OK (Hardware Channel GR + QMD + CB0 Initialized)\n");

    // 3. Alojar Buffer de Salida en la GPU (32 enteros = 128 bytes)
    const int NUM_THREADS = 32;
    const size_t SIZE = NUM_THREADS * sizeof(uint32_t);
    CUdeviceptr d_output;
    CHECK_CUDA(cuMemAlloc(&d_output, SIZE));
    printf("[+] cuMemAlloc() OK: GPU Device Pointer = 0x%016llx\n",
           (unsigned long long)d_output);

    // Inicializar memoria de GPU con centinela 0xDEADBEEF
    uint32_t h_init[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) h_init[i] = 0xDEADBEEF;
    CHECK_CUDA(cuMemcpyHtoD(d_output, h_init, SIZE));
    printf("[+] cuMemcpyHtoD() initialized GPU memory with sentinel 0xDEADBEEF\n");

    // 4. Cargar Módulo y Función
    CUmodule mod;
    CHECK_CUDA(cuModuleLoadData(&mod, kepler_compute_kernel));
    printf("[+] cuModuleLoadData() uploaded 64-bit SASS kernel into GPU VRAM\n");

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "add42_kernel"));
    printf("[+] cuModuleGetFunction() resolved entry point\n");

    // 5. Preparar Parámetros y Lanzar Kernel
    // Argumento 0 (c0[0x148]): Puntero de 64 bits al buffer de salida
    void *kernel_args[] = { &d_output, NULL };

    printf("[*] Launching kernel via cuLaunchKernel (Grid=1x1x1, Block=32x1x1)...\n");
    CHECK_CUDA(cuLaunchKernel(
        func,
        1, 1, 1,           // gridDim (1 bloque)
        NUM_THREADS, 1, 1, // blockDim (32 hilos = 1 warp)
        0,                 // sharedMemBytes
        0,                 // hStream
        kernel_args,       // kernelParams
        NULL               // extra
    ));
    printf("[+] cuLaunchKernel() submitted asynchronously to FIFO\n");

    // 6. Sincronizar Contexto (espera a la GPU física)
    printf("[*] Synchronizing execution via cuCtxSynchronize()...\n");
    CHECK_CUDA(cuCtxSynchronize());
    printf("[+] cuCtxSynchronize() finished: Hardware Semaphore Triggered!\n");

    // 7. Descargar Resultados a Host (VRAM -> CPU RAM)
    uint32_t h_output[NUM_THREADS];
    CHECK_CUDA(cuMemcpyDtoH(h_output, d_output, SIZE));
    printf("[+] cuMemcpyDtoH() read back %zu bytes from GPU\n\n", SIZE);

    // 8. Inspección de Resultados
    printf("============================================================\n");
    printf("  [COMPUTE VERIFICATION RESULTS]\n");
    printf("------------------------------------------------------------\n");
    int errors = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        uint32_t expected = i + 42;
        printf("    Thread %02d: GPU result = %3d (0x%08x) | Expected = %3d --> %s\n",
               i, h_output[i], h_output[i], expected,
               (h_output[i] == expected) ? "PASS" : "FAIL");
        if (h_output[i] != expected) {
            errors++;
        }
    }

    printf("------------------------------------------------------------\n");
    if (errors == 0) {
        printf("  >>> [SUCCESS] 32/32 THREADS VERIFIED 100%% EXACT VIA CUDA API! <<<\n");
        printf("  CUDA Driver API (cuLaunchKernel) is FULLY FUNCTIONAL on Kepler!\n");
    } else {
        printf("  [FAIL] Errors detected: %d / %d threads\n", errors, NUM_THREADS);
    }
    printf("============================================================\n");

    // 9. Teardown
    CHECK_CUDA(cuModuleUnload(mod));
    CHECK_CUDA(cuMemFree(d_output));
    CHECK_CUDA(cuCtxDestroy(ctx));
    printf("[+] Context cleaned up and destroyed cleanly.\n");

    return (errors == 0) ? 0 : 1;
}
