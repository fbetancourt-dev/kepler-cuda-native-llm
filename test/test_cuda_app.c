#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "[-] Error en %s: código %d\n", #call, err); \
        return 1; \
    } \
} while (0)

int main() {
    printf("============================================================\n");
    printf("  APLICACION CLIENTE CUDA (Usando libcuda.so nativo Nouveau)\n");
    printf("============================================================\n");

    // 1. Inicialización
    CHECK_CUDA(cuInit(0));
    printf("[+] cuInit() exitoso!\n");

    int driver_version = 0;
    CHECK_CUDA(cuDriverGetVersion(&driver_version));
    printf("[+] Versión del driver reportada: %d\n", driver_version);

    // 2. Consulta de dispositivos
    int device_count = 0;
    CHECK_CUDA(cuDeviceGetCount(&device_count));
    printf("[+] Dispositivos CUDA detectados: %d\n", device_count);

    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));

    char name[128];
    CHECK_CUDA(cuDeviceGetName(name, sizeof(name), dev));
    printf("[+] Dispositivo 0: %s\n", name);

    size_t total_mem = 0;
    CHECK_CUDA(cuDeviceTotalMem(&total_mem, dev));
    printf("[+] Memoria total de video: %zu MB (%zu bytes)\n", total_mem / (1024 * 1024), total_mem);

    int sm_count = 0, major = 0, minor = 0, clock_rate = 0;
    cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);

    printf("[+] Multiprocesadores SMX: %d (%d CUDA cores totales)\n", sm_count, sm_count * 192);
    printf("[+] Capacidad de cómputo: sm_%d%d\n", major, minor);
    printf("[+] Frecuencia de reloj: %d MHz\n", clock_rate / 1000);

    // 3. Crear Contexto CUDA
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));
    printf("[+] Contexto de ejecución CUDA creado en el hardware!\n");

    // 4. Reserva y transferencia de memoria en VRAM
    const int NUM_ELEMENTS = 256;
    const size_t BUFFER_SIZE = NUM_ELEMENTS * sizeof(float);
    float h_in[NUM_ELEMENTS];
    float h_out[NUM_ELEMENTS];

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_in[i] = (float)(i * 1.5f + 10.0f);
        h_out[i] = 0.0f;
    }

    CUdeviceptr d_mem;
    CHECK_CUDA(cuMemAlloc(&d_mem, BUFFER_SIZE));
    printf("[+] cuMemAlloc() asigno buffer en VRAM: dptr = 0x%llx (%zu bytes)\n",
           (unsigned long long)d_mem, BUFFER_SIZE);

    // Host -> Device (Subir a los chips GDDR5 de la Mac)
    CHECK_CUDA(cuMemcpyHtoD(d_mem, h_in, BUFFER_SIZE));
    printf("[+] cuMemcpyHtoD() transfirio %zu bytes de RAM de CPU -> VRAM GDDR5!\n", BUFFER_SIZE);

    // Device -> Host (Bajar de los chips GDDR5 a RAM)
    CHECK_CUDA(cuMemcpyDtoH(h_out, d_mem, BUFFER_SIZE));
    printf("[+] cuMemcpyDtoH() leyo %zu bytes de vuelta desde la GPU!\n", BUFFER_SIZE);

    // Validación de integridad
    int errors = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        if (h_out[i] != h_in[i]) {
            errors++;
        }
    }

    if (errors == 0) {
        printf("[OK] Verificación de integridad: 100%% EXACTA! (256/256 floats validados)\n");
        printf("     Ejemplo: h_in[0]=%.1f -> h_out[0]=%.1f, h_in[10]=%.1f -> h_out[10]=%.1f\n",
               h_in[0], h_out[0], h_in[10], h_out[10]);
    } else {
        printf("[-] Error de integridad: %d elementos no coinciden!\n", errors);
    }

    // 5. Liberación
    CHECK_CUDA(cuMemFree(d_mem));
    printf("[+] Memoria de dispositivo liberada.\n");

    CHECK_CUDA(cuCtxDestroy(ctx));
    printf("[+] Contexto CUDA destruido limpiamente.\n");

    printf("============================================================\n");
    printf("[OK] Todo el ciclo de vida de la CUDA Driver API funciono!\n");
    printf("============================================================\n");
    return 0;
}
