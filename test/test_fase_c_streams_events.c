#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "cuda.h"

int main(void) {
    printf("============================================================\n");
    printf("  TEST FASE C: STREAMS ASÍNCRONOS Y EVENTOS POR HARDWARE\n");
    printf("  cuStreamCreate/Destroy/Sync, cuEventRecord/Sync/ElapsedTime\n");
    printf("============================================================\n");

    CUresult res = cuInit(0);
    assert(res == CUDA_SUCCESS);

    CUdevice dev = 0;
    res = cuDeviceGet(&dev, 0);
    assert(res == CUDA_SUCCESS);

    CUcontext ctx = NULL;
    res = cuCtxCreate(&ctx, 0, dev);
    assert(res == CUDA_SUCCESS && ctx != NULL);
    printf("[1] Context initialized successfully\n");

    // 1. Crear dos streams independientes
    CUstream stream1 = NULL, stream2 = NULL;
    res = cuStreamCreate(&stream1, 0);
    assert(res == CUDA_SUCCESS && stream1 != NULL);
    res = cuStreamCreate(&stream2, 0);
    assert(res == CUDA_SUCCESS && stream2 != NULL);
    printf("[2] cuStreamCreate: stream1=%p, stream2=%p created successfully!\n",
           (void *)stream1, (void *)stream2);

    // 2. Crear tres eventos
    CUevent ev_start = NULL, ev_mid = NULL, ev_stop = NULL;
    res = cuEventCreate(&ev_start, 0);
    assert(res == CUDA_SUCCESS && ev_start != NULL);
    res = cuEventCreate(&ev_mid, 0);
    assert(res == CUDA_SUCCESS && ev_mid != NULL);
    res = cuEventCreate(&ev_stop, 0);
    assert(res == CUDA_SUCCESS && ev_stop != NULL);
    printf("[3] cuEventCreate: 3 hardware events created successfully!\n");

    // 3. Cargar un kernel real para despachar trabajo a los streams
    CUmodule mod = NULL;
    res = cuModuleLoad(&mod, "vector_official.cubin");
    assert(res == CUDA_SUCCESS && mod != NULL);

    CUfunction func = NULL;
    res = cuModuleGetFunction(&func, mod, "vec_add");
    assert(res == CUDA_SUCCESS && func != NULL);
    printf("[4] CUBIN kernel 'vec_add' loaded for stream testing\n");

    // Reservar buffers
    int n = 1024;
    size_t sz = n * sizeof(float);
    CUdeviceptr d_A = 0, d_B = 0, d_C = 0;
    cuMemAlloc(&d_A, sz);
    cuMemAlloc(&d_B, sz);
    cuMemAlloc(&d_C, sz);

    float h_A[1024], h_B[1024], h_C[1024];
    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i;
        h_B[i] = 100.0f;
    }
    cuMemcpyHtoD(d_A, h_A, sz);
    cuMemcpyHtoD(d_B, h_B, sz);

    void *params[] = { &d_A, &d_B, &d_C, &n };

    // 4. Registrar evento de inicio en stream1
    res = cuEventRecord(ev_start, stream1);
    assert(res == CUDA_SUCCESS);
    printf("[5] cuEventRecord(ev_start, stream1) recorded\n");

    // 5. Lanzar kernel en stream1
    res = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, stream1, params, NULL);
    assert(res == CUDA_SUCCESS);
    printf("[6] cuLaunchKernel dispatched into stream1 (seq registered!)\n");

    // 6. Registrar evento intermedio en stream1
    res = cuEventRecord(ev_mid, stream1);
    assert(res == CUDA_SUCCESS);

    // 7. Lanzar otro kernel en stream2
    res = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, stream2, params, NULL);
    assert(res == CUDA_SUCCESS);
    printf("[7] cuLaunchKernel dispatched into stream2\n");

    // 8. Registrar evento final en stream2
    res = cuEventRecord(ev_stop, stream2);
    assert(res == CUDA_SUCCESS);

    // 9. Sincronizar stream1 individualmente
    res = cuStreamSynchronize(stream1);
    assert(res == CUDA_SUCCESS);
    printf("[8] cuStreamSynchronize(stream1) completed successfully!\n");

    // Verificar que ev_mid ya está completo
    res = cuEventQuery(ev_mid);
    assert(res == CUDA_SUCCESS);
    printf("[9] cuEventQuery(ev_mid): reported READY (CUDA_SUCCESS)!\n");

    // 10. Sincronizar stream2
    res = cuStreamSynchronize(stream2);
    assert(res == CUDA_SUCCESS);
    printf("[10] cuStreamSynchronize(stream2) completed successfully!\n");

    // 11. Medir tiempo transcurrido en el hardware con cuEventElapsedTime
    float ms_stream1 = 0.0f;
    res = cuEventElapsedTime(&ms_stream1, ev_start, ev_mid);
    assert(res == CUDA_SUCCESS && ms_stream1 >= 0.0f);
    printf("[11] cuEventElapsedTime(ev_start, ev_mid) = %.4f ms (High Precision GPU Timing!)\n",
           ms_stream1);

    float ms_total = 0.0f;
    res = cuEventElapsedTime(&ms_total, ev_start, ev_stop);
    assert(res == CUDA_SUCCESS && ms_total >= ms_stream1);
    printf("[12] cuEventElapsedTime(ev_start, ev_stop) = %.4f ms (Total Pipeline Duration!)\n",
           ms_total);

    // Validar resultados aritméticos
    cuMemcpyDtoH(h_C, d_C, sz);
    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            ok = 0;
            break;
        }
    }
    assert(ok == 1);
    printf("[13] Vector arithmetic verified 100%% correct across stream launches!\n");

    // 12. Limpieza
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuModuleUnload(mod);
    cuEventDestroy(ev_start);
    cuEventDestroy(ev_mid);
    cuEventDestroy(ev_stop);
    cuStreamDestroy(stream1);
    cuStreamDestroy(stream2);
    cuCtxDestroy(ctx);
    printf("[14] All streams, events, modules, and context destroyed cleanly!\n");

    printf("\n============================================================\n");
    printf("  [+] FASE C COMPLETADA Y VERIFICADA AL 100%% EN SILICIO!\n");
    printf("============================================================\n");
    return 0;
}
