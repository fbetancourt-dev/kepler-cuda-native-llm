#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include "cuda.h"

static void *thread_test_func(void *arg) {
    (void)arg;
    CUcontext ctx = NULL;
    CUresult res = cuCtxGetCurrent(&ctx);
    assert(res == CUDA_SUCCESS);
    // En un nuevo hilo, g_current_ctx debe ser NULL debido a __thread
    printf("[TLS THREAD] New thread g_current_ctx = %p (Expected: NULL)\n", (void *)ctx);
    assert(ctx == NULL);
    return NULL;
}

int main(void) {
    printf("============================================================\n");
    printf("  TEST FASE A: BLINDAJE P0 DEL DRIVER CUDA KEPLER\n");
    printf("  Punteros interiores, ownership, versiones y TLS por hilo\n");
    printf("============================================================\n");

    CUresult res = cuInit(0);
    assert(res == CUDA_SUCCESS);

    int driverVer = 0;
    res = cuDriverGetVersion(&driverVer);
    assert(res == CUDA_SUCCESS);
    printf("[1] cuDriverGetVersion: %d (Expected: 10020)\n", driverVer);
    assert(driverVer == 10020);

    CUdevice dev = 0;
    res = cuDeviceGet(&dev, 0);
    assert(res == CUDA_SUCCESS);

    CUcontext ctx = NULL;
    res = cuCtxCreate(&ctx, 0, dev);
    assert(res == CUDA_SUCCESS && ctx != NULL);
    printf("[2] cuCtxCreate: context=%p created successfully\n", (void *)ctx);

    // Test TLS: crear un hilo secundario y verificar que no hereda accidentalmente el puntero crudo
    pthread_t th;
    pthread_create(&th, NULL, thread_test_func, NULL);
    pthread_join(th, NULL);
    printf("[3] TLS thread-local context isolation: PASSED!\n");

    // Test de asignación y puntero interior con offset
    size_t alloc_sz = 4096;
    CUdeviceptr d_buf = 0;
    res = cuMemAlloc(&d_buf, alloc_sz);
    assert(res == CUDA_SUCCESS && d_buf != 0);
    printf("[4] cuMemAlloc allocated 4096 bytes at GPU VA: 0x%lx\n", (unsigned long)d_buf);

    // Escribir en un offset interior (ej. +512 bytes)
    uint32_t host_src[64];
    for (int i = 0; i < 64; i++) host_src[i] = 0xbeef0000 + i;

    CUdeviceptr d_sub = d_buf + 512;
    res = cuMemcpyHtoD(d_sub, host_src, sizeof(host_src));
    assert(res == CUDA_SUCCESS);
    printf("[5] cuMemcpyHtoD to interior pointer (offset +512B): SUCCESS!\n");

    // Leer de vuelta del offset interior
    uint32_t host_dst[64];
    memset(host_dst, 0, sizeof(host_dst));
    res = cuMemcpyDtoH(host_dst, d_sub, sizeof(host_dst));
    assert(res == CUDA_SUCCESS);
    for (int i = 0; i < 64; i++) {
        assert(host_dst[i] == 0xbeef0000 + i);
    }
    printf("[6] cuMemcpyDtoH from interior pointer (offset +512B): 64/64 words match perfectly!\n");

    // Intentar cuMemFree en el puntero interior (debe ser rechazado con CUDA_ERROR_INVALID_VALUE)
    res = cuMemFree(d_sub);
    printf("[7] cuMemFree(interior_pointer) returned: %d (Expected: %d CUDA_ERROR_INVALID_VALUE)\n",
           res, CUDA_ERROR_INVALID_VALUE);
    assert(res == CUDA_ERROR_INVALID_VALUE);

    // cuMemFree en la dirección base correcta debe ser exitoso
    res = cuMemFree(d_buf);
    assert(res == CUDA_SUCCESS);
    printf("[8] cuMemFree(base_pointer) freed cleanly: SUCCESS!\n");

    // Test de límites de lanzamiento en cuLaunchKernel
    // Usamos un puntero no nulo simulado para evaluar la validación de dimensiones
    CUfunction dummy_func = (CUfunction)(uintptr_t)0xdeadbeef;
    res = cuLaunchKernel(dummy_func, 0, 1, 1, 128, 1, 1, 0, NULL, NULL, NULL);
    assert(res == CUDA_ERROR_INVALID_VALUE);
    printf("[9] cuLaunchKernel dimension 0 rejection: PASSED (Returned CUDA_ERROR_INVALID_VALUE)\n");

    res = cuLaunchKernel(dummy_func, 1, 1, 1, 1025, 1, 1, 0, NULL, NULL, NULL);
    assert(res == CUDA_ERROR_INVALID_VALUE);
    printf("[10] cuLaunchKernel threads > 1024 rejection: PASSED (Returned CUDA_ERROR_INVALID_VALUE)\n");

    res = cuCtxDestroy(ctx);
    assert(res == CUDA_SUCCESS);
    printf("[11] cuCtxDestroy: context cleaned up successfully!\n");

    printf("\n============================================================\n");
    printf("  [+] FASE A COMPLETADA Y VERIFICADA AL 100%% EN SILICIO!\n");
    printf("============================================================\n");
    return 0;
}
