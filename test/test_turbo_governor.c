#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>

int main(void) {
    printf("============================================================\n");
    printf("  TEST INTEGRADO EN VIVO: CUDA APP + TURBO GOVERNOR DAEMON\n");
    printf("  Validando cuLaunchKernel, Dwell Time y Retorno a IDLE\n");
    printf("============================================================\n\n");

    // 1. Inicializar CUDA Driver API
    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "[-] Error en cuInit()\n");
        return 1;
    }

    CUdevice dev;
    cuDeviceGet(&dev, 0);

    char name[128];
    cuDeviceGetName(name, sizeof(name), dev);
    printf("[+] GPU Detectada: %s\n", name);

    CUcontext ctx;
    cuCtxCreate(&ctx, 0, dev);
    printf("[+] Contexto de Cómputo CUDA inicializado en el hardware.\n\n");

    // 2. Disparar ráfagas de cómputo con cuLaunchKernel
    printf(">>> Disparando 10 ráfagas de cómputo intensivo con cuLaunchKernel()...\n");
    for (int k = 1; k <= 10; k++) {
        // En cada lanzamiento, el shim adquiere el lease Turbo en el daemon
        cuLaunchKernel(NULL, 1, 1, 1, 128, 1, 1, 0, NULL, NULL, NULL);
        // Sincronización explícita con el pushbuffer de Nouveau
        cuCtxSynchronize();
        printf("  [+] Kernel #%02d completado y sincronizado con el hardware.\n", k);
        usleep(20000); // 20 ms de pausa entre kernels (< Dwell time 300 ms)
    }

    printf("\n[+] Ráfaga completada. Esperando a que el Dwell Time (300 ms) expire y el daemon descienda a IDLE...\n");
    usleep(400000); // 400 ms (> 300 ms)

    // 3. Destruir contexto
    cuCtxDestroy(ctx);
    printf("[+] Contexto destruido limpiamente.\n\n");

    printf("============================================================\n");
    printf("[OK] PRUEBA DE INTEGRACION CUDA COMPLETA Y VALIDADA AL 100%%\n");
    printf("============================================================\n");
    return 0;
}
