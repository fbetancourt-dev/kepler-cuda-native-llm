#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "cuda.h"

int main(void) {
    printf("============================================================\n");
    printf("  TEST FASE B: RUNTIME DRIVER APIS (GGML / LLAMA.CPP / BENCH)\n");
    printf("  MemGetInfo, AllocPitch, PointerAttributes, DtoD, Attributes\n");
    printf("============================================================\n");

    CUresult res = cuInit(0);
    assert(res == CUDA_SUCCESS);

    CUdevice dev = 0;
    res = cuDeviceGet(&dev, 0);
    assert(res == CUDA_SUCCESS);

    // 1. Validar matriz de atributos de dispositivo para sm_30
    int sm_count = 0, cc_major = 0, cc_minor = 0, max_regs = 0, max_threads_sm = 0, l2_sz = 0;
    cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&max_regs, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, dev);
    cuDeviceGetAttribute(&max_threads_sm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    cuDeviceGetAttribute(&l2_sz, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

    printf("[1] Device Attributes: SMs=%d (GT 750M), CC=%d.%d, Regs/SM=%d, Threads/SM=%d, L2=%d KB\n",
           sm_count, cc_major, cc_minor, max_regs, max_threads_sm, l2_sz / 1024);
    assert(sm_count == 2 && cc_major == 3 && cc_minor == 0);
    assert(max_regs == 65536 && max_threads_sm == 2048 && l2_sz == 262144);

    CUcontext ctx = NULL;
    res = cuCtxCreate(&ctx, 0, dev);
    assert(res == CUDA_SUCCESS && ctx != NULL);

    // 2. cuMemGetInfo_v2
    size_t free_mem = 0, total_mem = 0;
    res = cuMemGetInfo_v2(&free_mem, &total_mem);
    assert(res == CUDA_SUCCESS && total_mem > 0 && free_mem > 0);
    printf("[2] cuMemGetInfo_v2: Total = %zu MB (%.2f GB), Free = %zu MB\n",
           total_mem / (1024 * 1024), (double)total_mem / (1024.0 * 1024.0 * 1024.0),
           free_mem / (1024 * 1024));
    assert(free_mem <= total_mem);

    // 3. cuMemAllocPitch (ej. imagen/matriz 1000 bytes de ancho x 100 filas)
    CUdeviceptr d_pitch_buf = 0;
    size_t pitch = 0;
    res = cuMemAllocPitch(&d_pitch_buf, &pitch, 1000, 100, 4);
    assert(res == CUDA_SUCCESS && d_pitch_buf != 0);
    printf("[3] cuMemAllocPitch: Width=1000, Height=100 -> Pitch=%zu bytes (Expected 256B aligned: %zu)\n",
           pitch, (1000 + 255) & ~255);
    assert(pitch == ((1000 + 255) & ~255));
    assert((pitch % 256) == 0);

    // 4. cuMemGetAddressRange
    CUdeviceptr pbase = 0;
    size_t psize = 0;
    CUdeviceptr interior_ptr = d_pitch_buf + 500;
    res = cuMemGetAddressRange(&pbase, &psize, interior_ptr);
    assert(res == CUDA_SUCCESS);
    printf("[4] cuMemGetAddressRange: Queried 0x%lx -> Base=0x%lx, Size=%zu\n",
           (unsigned long)interior_ptr, (unsigned long)pbase, psize);
    assert(pbase == d_pitch_buf);
    assert(psize == pitch * 100);

    // 5. cuPointerGetAttribute
    CUcontext ptr_ctx = NULL;
    CUmemorytype mem_type = 0;
    res = cuPointerGetAttribute(&ptr_ctx, CU_POINTER_ATTRIBUTE_CONTEXT, interior_ptr);
    assert(res == CUDA_SUCCESS && ptr_ctx == ctx);

    res = cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, interior_ptr);
    assert(res == CUDA_SUCCESS && mem_type == CU_MEMORYTYPE_DEVICE);
    printf("[5] cuPointerGetAttribute: Context match (%p), MemoryType = DEVICE (%d): PASSED!\n",
           (void *)ptr_ctx, mem_type);

    // 6. cuMemcpyDtoD (Device to Device Copy in VRAM)
    CUdeviceptr d_src = 0, d_dst = 0;
    res = cuMemAlloc(&d_src, 1024 * sizeof(float));
    assert(res == CUDA_SUCCESS);
    res = cuMemAlloc(&d_dst, 1024 * sizeof(float));
    assert(res == CUDA_SUCCESS);

    float host_in[1024];
    for (int i = 0; i < 1024; i++) host_in[i] = (float)(i * 3.14159f);

    res = cuMemcpyHtoD(d_src, host_in, sizeof(host_in));
    assert(res == CUDA_SUCCESS);

    // Copiar Device a Device internamente en la GPU
    res = cuMemcpyDtoD(d_dst, d_src, sizeof(host_in));
    assert(res == CUDA_SUCCESS);
    printf("[6] cuMemcpyDtoD: Transferred 4096 bytes directly across VRAM buffers: SUCCESS!\n");

    // Verificar en CPU que d_dst tenga exactamente los datos de d_src
    float host_out[1024];
    memset(host_out, 0, sizeof(host_out));
    res = cuMemcpyDtoH(host_out, d_dst, sizeof(host_out));
    assert(res == CUDA_SUCCESS);

    int d2d_match = 1;
    for (int i = 0; i < 1024; i++) {
        if (host_out[i] != host_in[i]) {
            d2d_match = 0;
            printf("Mismatch at %d: out=%f, expected=%f\n", i, host_out[i], host_in[i]);
            break;
        }
    }
    assert(d2d_match == 1);
    printf("[7] cuMemcpyDtoD Data Verification: 1024/1024 floats bitwise identical on CPU readback!\n");

    // Limpieza
    cuMemFree(d_pitch_buf);
    cuMemFree(d_src);
    cuMemFree(d_dst);
    cuCtxDestroy(ctx);

    printf("\n============================================================\n");
    printf("  [+] FASE B COMPLETADA Y VERIFICADA AL 100%% EN SILICIO!\n");
    printf("============================================================\n");
    return 0;
}
