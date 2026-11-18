#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <nouveau/nouveau.h>

#include "drf.h"
#include "cla0c0qmd.h"

#define KEPLER_COMPUTE_A           0xa0c0

#define NVA0C0_QMDV00_06_VAL_SET(p,a...) NVVAL_MW_SET((p), NVA0C0, QMDV00_06, ##a)
#define NVA0C0_QMDV00_06_DEF_SET(p,a...) NVDEF_MW_SET((p), NVA0C0, QMDV00_06, ##a)

// Formato canónico de paquetes de pushbuffer para Fermi / Kepler (NVC0+)
static inline uint32_t NVC0_FIFO_PKHDR_SQ(int subc, int mthd, unsigned size) {
    return 0x20000000 | (size << 16) | (subc << 13) | (mthd >> 2);
}

// Métodos de Kepler Compute (clase 0xa0c0 / NVE4)
#define NVE4_COMPUTE_SERIALIZE         0x0110
#define NVE4_COMPUTE_SHARED_BASE       0x0214
#define NVE4_COMPUTE_LAUNCH_DESC_ADDR  0x02b4
#define NVE4_COMPUTE_LAUNCH            0x02bc
#define NVE4_COMPUTE_UNK0310           0x0310
#define NVE4_COMPUTE_LOCAL_BASE        0x077c
#define NVE4_COMPUTE_CODE_ADDRESS_HIGH 0x1608
#define NVE4_COMPUTE_CODE_ADDRESS_LOW  0x160c
#define NVE4_COMPUTE_FLUSH             0x1698
#define NVE4_COMPUTE_QUERY_ADDR_HIGH   0x1b00
#define NVE4_COMPUTE_QUERY_ADDR_LOW    0x1b04
#define NVE4_COMPUTE_QUERY_SEQUENCE    0x1b08
#define NVE4_COMPUTE_QUERY_GET         0x1b0c

int main(void) {
    printf("============================================================\n");
    printf("  NVIDIA Kepler GK107 - Live Mathematical Compute in Silicon\n");
    printf("  Warp of 32 Threads: output[i] = threadIdx.x + 42\n");
    printf("  Class 0xa0c0 + CB0 (c[0x0][0x148]) + SASS ST.E 64-bit DMA\n");
    printf("============================================================\n\n");

    const char *nodes[] = { "/dev/dri/renderD128", "/dev/dri/card0", NULL };
    int fd = -1;
    for (int i = 0; nodes[i] != NULL; i++) {
        fd = open(nodes[i], O_RDWR);
        if (fd >= 0) break;
    }
    if (fd < 0) {
        perror("[-] Failed to open DRM device");
        return 1;
    }

    struct nouveau_device *dev = NULL;
    int ret = nouveau_device_wrap(fd, 0, &dev);
    if (ret != 0 || !dev) {
        fprintf(stderr, "[-] nouveau_device_wrap failed: %d\n", ret);
        close(fd);
        return 1;
    }
    printf("[+] nouveau_device_wrap() OK (chipset: 0x%02x - GK107 Kepler)\n", dev->chipset);

    struct nouveau_client *client = NULL;
    ret = nouveau_client_new(dev, &client);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_client_new failed: %d\n", ret);
        return 1;
    }

    struct nouveau_object *chan = NULL;
    struct nve0_fifo fifo_args = { .engine = NVE0_FIFO_ENGINE_GR };
    ret = nouveau_object_new(&dev->object, 0, NOUVEAU_FIFO_CHANNEL_CLASS,
                             &fifo_args, sizeof(fifo_args), &chan);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_object_new(FIFO_CHANNEL) failed: %d\n", ret);
        return 1;
    }
    printf("[+] Hardware FIFO channel allocated successfully.\n");

    struct nouveau_object *compute_obj = NULL;
    ret = nouveau_object_new(chan, 0xbeef0001, KEPLER_COMPUTE_A, NULL, 0, &compute_obj);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_object_new(KEPLER_COMPUTE_A) failed: %d\n", ret);
        return 1;
    }
    printf("[+] Object 0xbeef0001 instantiated with class 0x%04x (KEPLER_COMPUTE_A).\n", KEPLER_COMPUTE_A);

    struct nouveau_pushbuf *push = NULL;
    ret = nouveau_pushbuf_new(client, chan, 4, 32 * 1024, true, &push);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_pushbuf_new failed: %d\n", ret);
        return 1;
    }

    // 1. Asignar Output Buffer en GART (accessible concurrentemente por GPU y CPU)
    struct nouveau_bo *output_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_GART, 256, 4096, NULL, &output_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate output_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(output_bo, NOUVEAU_BO_RDWR, client);
    volatile uint32_t *output_ptr = (volatile uint32_t *)output_bo->map;
    for (int i = 0; i < 64; i++) {
        output_ptr[i] = 0xDEADBEEF; // Centinela antes de que la GPU escriba
    }
    printf("[+] Output Data Buffer allocated at GART GPU VA: 0x%016llx\n",
           (unsigned long long)output_bo->offset);

    // 2. Asignar Constant Buffer 0 (CB0) en VRAM (1024 bytes) para parámetros del kernel
    struct nouveau_bo *cb0_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_VRAM, 256, 1024, NULL, &cb0_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate cb0_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(cb0_bo, NOUVEAU_BO_WR, client);
    memset(cb0_bo->map, 0, 1024);
    // Escribir el puntero de 64 bits de output_bo en CB0 offset 0x148 (convención sm_30 ABI)
    uint32_t *cb0_ptr = (uint32_t *)cb0_bo->map;
    cb0_ptr[0x148 / 4]     = (uint32_t)(output_bo->offset & 0xffffffff);
    cb0_ptr[(0x148 / 4) + 1] = (uint32_t)(output_bo->offset >> 32);
    printf("[+] Constant Buffer 0 (CB0) configured at VRAM: 0x%016llx\n",
           (unsigned long long)cb0_bo->offset);
    printf("    [c0[0x148]]: Output Pointer Low  = 0x%08x\n", cb0_ptr[0x148 / 4]);
    printf("    [c0[0x14c]]: Output Pointer High = 0x%08x\n", cb0_ptr[(0x148 / 4) + 1]);

    // 3. Asignar Buffer de Código VRAM y subir el SASS Math Microkernel (64 bytes)
    struct nouveau_bo *code_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_VRAM, 256, 4096, NULL, &code_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate code_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(code_bo, NOUVEAU_BO_WR, client);
    uint32_t *code_ptr = (uint32_t *)code_bo->map;
    memset(code_ptr, 0, 4096);

    // Kepler SW Sched Bundle de 64 bytes:
    // Offset 0x00: Control Word de Scheduling
    code_ptr[0] = (uint32_t)(0x00000007 | (0x24 << 4) | (0x24 << 12) | (0x24 << 20) | (0x2eULL << 28));
    code_ptr[1] = 0x20000000 | (0x2e >> 4);
    // Offset 0x08: ld b32 $r2 c0[0x148]  |  ld b32 $r3 c0[0x14c]
    code_ptr[2] = 0x53f09e1c;
    code_ptr[3] = 0x53f0df1c;
    // Offset 0x10: mov b32 $r0 $tidx     |  shl b32 $r4 $r0 2
    code_ptr[4] = 0x42101c08;
    code_ptr[5] = 0x08011e08;
    // Offset 0x18: add b32 $r2 $r2 $r4   |  add b32 $r5 $r0 42 (0x2a)
    code_ptr[6] = 0x10209c2c;
    code_ptr[7] = 0xa8015cac;
    // Offset 0x20: st b32 g[$r2d] $r5    |  exit
    code_ptr[8] = 0x00215c5b;
    code_ptr[9] = 0x88001dff;
    // Offset 0x28..0x3C: 0 (NOPs)

    printf("[+] SASS Math Microkernel uploaded to VRAM at offset 0x%016llx\n",
           (unsigned long long)code_bo->offset);

    // 4. Buffer de Query de Hardware (Semáforo de sincronización) en GART
    struct nouveau_bo *query_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_GART, 256, 4096, NULL, &query_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate query_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(query_bo, NOUVEAU_BO_RDWR, client);
    volatile uint32_t *query_ptr = (volatile uint32_t *)query_bo->map;
    query_ptr[0] = 0xDEADBEEF;

    // 5. Configurar Descriptor QMD v00_06 (256 bytes)
    struct nouveau_bo *desc_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_VRAM, 256, 256, NULL, &desc_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate desc_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(desc_bo, NOUVEAU_BO_WR, client);
    uint32_t *qmd = (uint32_t *)desc_bo->map;
    memset(qmd, 0, 256);

    NVA0C0_QMDV00_06_DEF_SET(qmd, INVALIDATE_TEXTURE_HEADER_CACHE, TRUE);
    NVA0C0_QMDV00_06_DEF_SET(qmd, INVALIDATE_TEXTURE_SAMPLER_CACHE, TRUE);
    NVA0C0_QMDV00_06_DEF_SET(qmd, INVALIDATE_TEXTURE_DATA_CACHE, TRUE);
    NVA0C0_QMDV00_06_DEF_SET(qmd, INVALIDATE_SHADER_DATA_CACHE, TRUE);
    NVA0C0_QMDV00_06_DEF_SET(qmd, INVALIDATE_INSTRUCTION_CACHE, TRUE);
    NVA0C0_QMDV00_06_DEF_SET(qmd, INVALIDATE_SHADER_CONSTANT_CACHE, TRUE);
    NVA0C0_QMDV00_06_DEF_SET(qmd, RELEASE_MEMBAR_TYPE, FE_SYSMEMBAR);
    NVA0C0_QMDV00_06_DEF_SET(qmd, CWD_MEMBAR_TYPE, L1_SYSMEMBAR);
    NVA0C0_QMDV00_06_DEF_SET(qmd, API_VISIBLE_CALL_LIMIT, NO_CHECK);
    NVA0C0_QMDV00_06_VAL_SET(qmd, SASS_VERSION, 0x30); // Kepler SM 3.0

    NVA0C0_QMDV00_06_VAL_SET(qmd, PROGRAM_OFFSET, 0);

    // Geometría: 1 bloque x 32 hilos (1 warp completo)
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_WIDTH, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_HEIGHT, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_DEPTH, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION0, 32);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION1, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION2, 1);

    NVA0C0_QMDV00_06_DEF_SET(qmd, L1_CONFIGURATION, DIRECTLY_ADDRESSABLE_MEMORY_SIZE_16KB);
    NVA0C0_QMDV00_06_VAL_SET(qmd, REGISTER_COUNT, 16);
    NVA0C0_QMDV00_06_VAL_SET(qmd, BARRIER_COUNT, 0);

    // Enlazar Constant Buffer 0 (CB0) en el QMD (index = 0)
    NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_ADDR_LOWER, 0, (uint32_t)(cb0_bo->offset & 0xffffffff));
    NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_ADDR_UPPER, 0, (uint32_t)(cb0_bo->offset >> 32));
    NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_SIZE, 0, 1024);
    NVA0C0_QMDV00_06_DEF_SET(qmd, CONSTANT_BUFFER_VALID, 0, TRUE);

    // Semáforo de hardware al terminar el kernel
    NVA0C0_QMDV00_06_DEF_SET(qmd, SEMAPHORE_RELEASE_ENABLE0, TRUE);
    NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_ADDRESS_LOWER, (uint32_t)(query_bo->offset & 0xffffffff));
    NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_ADDRESS_UPPER, (uint32_t)(query_bo->offset >> 32));
    NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_PAYLOAD, 0xCAFEBABE);
    NVA0C0_QMDV00_06_DEF_SET(qmd, RELEASE0_STRUCTURE_SIZE, ONE_WORD);

    printf("[+] QMD v00_06 configured with Constant Buffer 0 and Hardware Semaphore.\n");

    // 6. Reservar espacio en pushbuffer y registrar referencias a los 5 BOs
    ret = nouveau_pushbuf_space(push, 128, 0, 0);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_pushbuf_space failed: %d\n", ret);
        return 1;
    }

    struct nouveau_pushbuf_refn refs[] = {
        { code_bo,   NOUVEAU_BO_VRAM | NOUVEAU_BO_RD },
        { cb0_bo,    NOUVEAU_BO_VRAM | NOUVEAU_BO_RD },
        { desc_bo,   NOUVEAU_BO_VRAM | NOUVEAU_BO_RD },
        { output_bo, NOUVEAU_BO_GART | NOUVEAU_BO_WR },
        { query_bo,  NOUVEAU_BO_GART | NOUVEAU_BO_WR },
    };
    ret = nouveau_pushbuf_refn(push, refs, 5);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_pushbuf_refn failed: %d\n", ret);
        return 1;
    }

    // 7. Emitir métodos al pushbuffer
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, 0x0000, 1);
    *push->cur++ = compute_obj->oclass;

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LOCAL_BASE, 1);
    *push->cur++ = 0xff << 24;
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_SHARED_BASE, 1);
    *push->cur++ = 0xfe << 24;
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_UNK0310, 1);
    *push->cur++ = 0x300;

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_CODE_ADDRESS_HIGH, 2);
    *push->cur++ = (uint32_t)(code_bo->offset >> 32);
    *push->cur++ = (uint32_t)(code_bo->offset & 0xffffffff);

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_FLUSH, 1);
    *push->cur++ = 0x00000001; // FLUSH CODE

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LAUNCH_DESC_ADDR, 1);
    *push->cur++ = (uint32_t)(desc_bo->offset >> 8);

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LAUNCH, 1);
    *push->cur++ = 0x00000003;

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_SERIALIZE, 1);
    *push->cur++ = 0x00000000;

    printf("[*] Submitting compute warp via nouveau_pushbuf_kick()...\n");
    ret = nouveau_pushbuf_kick(push, chan);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_pushbuf_kick failed: %d\n", ret);
        return 1;
    }

    // 8. Esperar finalización en silicio
    printf("[*] Awaiting GPU completion via nouveau_bo_wait()...\n");
    ret = nouveau_bo_wait(query_bo, NOUVEAU_BO_RDWR, client);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_bo_wait failed: %d\n", ret);
        return 1;
    }

    printf("\n============================================================\n");
    printf("  [SILICON FEEDBACK RESULTS]\n");
    printf("  - QMD Hardware Release Token: 0x%08x\n", query_ptr[0]);
    printf("------------------------------------------------------------\n");

    int errors = 0;
    printf("  Inspecting 32 threads output array [output[i] = i + 42]:\n");
    for (int i = 0; i < 32; i++) {
        uint32_t val = output_ptr[i];
        uint32_t expected = i + 42;
        printf("    Thread %02d: GPU wrote %3d | Expected %3d --> %s\n",
               i, val, expected, (val == expected) ? "PASS" : "FAIL");
        if (val != expected) {
            errors++;
        }
    }

    printf("------------------------------------------------------------\n");
    if (query_ptr[0] == 0xCAFEBABE && errors == 0) {
        printf("  >>> [SUCCESS] 32/32 THREADS PRODUCED 100%% EXACT MATH! <<<\n");
        printf("  NVIDIA GeForce GT 750M (GK107) calculated in real silicon!\n");
        printf("  Constant Buffer 0 + SASS Instructions + Global DMA: VERIFIED.\n");
    } else {
        printf("  [NOTICE] Errors: %d. Check register mapping or MMU faults.\n", errors);
    }
    printf("============================================================\n");

    // Limpieza
    nouveau_bo_ref(NULL, &code_bo);
    nouveau_bo_ref(NULL, &cb0_bo);
    nouveau_bo_ref(NULL, &desc_bo);
    nouveau_bo_ref(NULL, &output_bo);
    nouveau_bo_ref(NULL, &query_bo);
    nouveau_pushbuf_del(&push);
    nouveau_object_del(&compute_obj);
    nouveau_object_del(&chan);
    nouveau_client_del(&client);
    nouveau_device_del(&dev);
    close(fd);

    return (errors == 0 && query_ptr[0] == 0xCAFEBABE) ? 0 : 1;
}
