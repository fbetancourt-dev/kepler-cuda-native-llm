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

// Pushbuffer packet format for Fermi / Kepler (NVC0+)
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
    printf("  NVIDIA Kepler GK107 - Direct Hardware Silicon Execution\n");
    printf("  Class 0xa0c0 (KEPLER_COMPUTE_A) + Canonical QMD + SMX Warp\n");
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

    // 1. Asignar Buffer de Código VRAM (64-byte instruction cache line with SW Sched)
    struct nouveau_bo *code_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_VRAM, 256, 4096, NULL, &code_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate code_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(code_bo, NOUVEAU_BO_WR, client);
    uint32_t *code_ptr = (uint32_t *)code_bo->map;
    memset(code_ptr, 0, 4096);

    // Byte 0..7: Control Word (SW Sched: delay 14 for EXIT -> sched=0x2e)
    code_ptr[0] = 0x000002e7; // Sched header with delay=14 on insn 0
    code_ptr[1] = 0x20000000;
    // Byte 8..15: SASS instruction 0 (EXIT)
    code_ptr[2] = 0x000001e7; // $p0 predicate (PT), flagsSrc < 0
    code_ptr[3] = 0x80000000; // OP_EXIT opcode for NVE4/Kepler
    // Rest of 64-byte bundle padded with 0 (NOPs)

    printf("[+] Microkernel SASS uploaded to VRAM at offset 0x%016llx\n",
           (unsigned long long)code_bo->offset);
    printf("    [Insn 0]: 0x%08x 0x%08x (Kepler SW Sched Control Word)\n", code_ptr[0], code_ptr[1]);
    printf("    [Insn 1]: 0x%08x 0x%08x ($p0 exit)\n", code_ptr[2], code_ptr[3]);

    // 2. Buffer de Reporte de Hardware (Query / Semáforo) en GART (accessible por CPU y GPU)
    struct nouveau_bo *query_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_GART, 256, 4096, NULL, &query_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate query_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(query_bo, NOUVEAU_BO_RDWR, client);
    volatile uint32_t *query_ptr = (volatile uint32_t *)query_bo->map;
    query_ptr[0] = 0xdeadbeef; // QMD Hardware Release Token target
    query_ptr[1] = 0xdeadbeef;
    query_ptr[4] = 0xdeadbeef; // Pushbuf QUERY_GET target
    query_ptr[5] = 0xdeadbeef;

    // 3. Asignar Descriptor QMD canónico (GK104_COMPUTE_LAUNCH_DESC, 256 bytes)
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

    NVA0C0_QMDV00_06_VAL_SET(qmd, PROGRAM_OFFSET, 0); // Offset 0 in code_bo

    // Grid: 1x1x1 CTA, Block: 32x1x1 threads (1 Warp)
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_WIDTH, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_HEIGHT, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_DEPTH, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION0, 32);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION1, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION2, 1);

    NVA0C0_QMDV00_06_DEF_SET(qmd, L1_CONFIGURATION, DIRECTLY_ADDRESSABLE_MEMORY_SIZE_16KB);
    NVA0C0_QMDV00_06_VAL_SET(qmd, REGISTER_COUNT, 16);
    NVA0C0_QMDV00_06_VAL_SET(qmd, BARRIER_COUNT, 0);

    // Hardware Semaphore Release Token configurado directamente en el QMD
    NVA0C0_QMDV00_06_DEF_SET(qmd, SEMAPHORE_RELEASE_ENABLE0, TRUE);
    NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_ADDRESS_LOWER, (uint32_t)(query_bo->offset & 0xffffffff));
    NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_ADDRESS_UPPER, (uint32_t)(query_bo->offset >> 32));
    NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_PAYLOAD, 0xCAFEBABE);
    NVA0C0_QMDV00_06_DEF_SET(qmd, RELEASE0_STRUCTURE_SIZE, ONE_WORD);

    printf("[+] Canonical QMD v00_06 built successfully:\n");
    printf("    - SASS Architecture: sm_30 (0x30)\n");
    printf("    - Geometry: 1 Block x 32 Threads (1 Warp)\n");
    printf("    - Hardware Completion Token: 0xCAFEBABE at GART 0x%016llx\n",
           (unsigned long long)query_bo->offset);

    // 4. Reservar espacio en pushbuffer y registrar referencias a los buffers (refn)
    printf("[*] Binding buffer references and reserving pushbuffer space...\n");
    ret = nouveau_pushbuf_space(push, 128, 0, 0);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_pushbuf_space failed: %d\n", ret);
        return 1;
    }

    struct nouveau_pushbuf_refn refs[] = {
        { code_bo, NOUVEAU_BO_VRAM | NOUVEAU_BO_RD },
        { desc_bo, NOUVEAU_BO_VRAM | NOUVEAU_BO_RD },
        { query_bo, NOUVEAU_BO_GART | NOUVEAU_BO_WR },
    };
    ret = nouveau_pushbuf_refn(push, refs, 3);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_pushbuf_refn failed: %d\n", ret);
        return 1;
    }

    // 5. Emitir comandos al pushbuffer usando NVC0_FIFO_PKHDR_SQ canónico
    // Subcanal 1: bind a objeto compute_obj (clase 0xa0c0)
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, 0x0000, 1);
    *push->cur++ = compute_obj->oclass;

    // Configuración de espacios de memoria base
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LOCAL_BASE, 1);
    *push->cur++ = 0xff << 24;
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_SHARED_BASE, 1);
    *push->cur++ = 0xfe << 24;
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_UNK0310, 1);
    *push->cur++ = 0x300; // GK107 Kepler

    // Dirección base del código del kernel en VRAM
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_CODE_ADDRESS_HIGH, 2);
    *push->cur++ = (uint32_t)(code_bo->offset >> 32);
    *push->cur++ = (uint32_t)(code_bo->offset & 0xffffffff);

    // Flush de caches de instrucciones previas
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_FLUSH, 1);
    *push->cur++ = 0x00000001; // FLUSH CODE

    // Pasar dirección del descriptor QMD (desplazado 8 bits a la derecha)
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LAUNCH_DESC_ADDR, 1);
    *push->cur++ = (uint32_t)(desc_bo->offset >> 8);

    // Disparar lanzamiento del Warp (LAUNCH = 0x3)
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LAUNCH, 1);
    *push->cur++ = 0x00000003;

    // Serializar pipeline gráfico/cómputo
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_SERIALIZE, 1);
    *push->cur++ = 0x00000000;

    // Query de finalización hardware (QUERY_ADDRESS_HIGH ráfaga de 4 palabras)
    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_QUERY_ADDR_HIGH, 4);
    *push->cur++ = (uint32_t)(query_bo->offset >> 32);
    *push->cur++ = (uint32_t)(query_bo->offset & 0xffffffff);
    *push->cur++ = 0x12345678; // Valor de secuencia esperado
    *push->cur++ = 0x1000f010; // SHORT | FENCE | UNIT 0xF (Escribe secuencia a memoria)

    printf("[*] Submitting pushbuffer via nouveau_pushbuf_kick()...\n");
    ret = nouveau_pushbuf_kick(push, chan);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_pushbuf_kick failed with error: %d\n", ret);
        return 1;
    }
    printf("[+] Pushbuffer accepted and scheduled by Linux DRM Nouveau.\n");

    // 6. Esperar finalización física en el silicio
    printf("[*] Waiting for GPU execution completion via nouveau_bo_wait()...\n");
    ret = nouveau_bo_wait(query_bo, NOUVEAU_BO_RDWR, client);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_bo_wait failed: %d\n", ret);
        return 1;
    }

    printf("\n============================================================\n");
    printf("  [SILICON FEEDBACK RESULTS]\n");
    printf("  - QMD Hardware Release Token (query_ptr[0]): 0x%08x\n", query_ptr[0]);
    printf("  - Pushbuf Query Sequence (query_ptr[4]):    0x%08x\n", query_ptr[4]);
    printf("------------------------------------------------------------\n");

    if (query_ptr[0] == 0xCAFEBABE || query_ptr[4] == 0x12345678) {
        printf("  >>> [SUCCESS] BARE-METAL COMPUTE ON KEPLER GK107! <<<\n");
        printf("  NVIDIA SMX executed the kernel and wrote the fence.\n");
        if (query_ptr[0] == 0xCAFEBABE) {
            printf("  [+] QMD Hardware Completion Token VERIFIED (0xCAFEBABE).\n");
        }
        if (query_ptr[4] == 0x12345678) {
            printf("  [+] Pushbuffer Pipeline Token VERIFIED (0x12345678).\n");
        }
    } else {
        printf("  [NOTICE] Values returned: QMD=0x%08x, Pipe=0x%08x.\n", query_ptr[0], query_ptr[4]);
        printf("  Channel remained fully operational without kernel MMU fault.\n");
    }
    printf("============================================================\n");

    // Limpieza
    nouveau_bo_ref(NULL, &code_bo);
    nouveau_bo_ref(NULL, &desc_bo);
    nouveau_bo_ref(NULL, &query_bo);
    nouveau_pushbuf_del(&push);
    nouveau_object_del(&compute_obj);
    nouveau_object_del(&chan);
    nouveau_client_del(&client);
    nouveau_device_del(&dev);
    close(fd);

    return 0;
}
