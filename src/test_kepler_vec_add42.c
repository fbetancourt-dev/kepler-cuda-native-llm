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

static inline uint32_t NVC0_FIFO_PKHDR_SQ(int subc, int mthd, unsigned size) {
    return 0x20000000 | (size << 16) | (subc << 13) | (mthd >> 2);
}

#define NVE4_COMPUTE_SERIALIZE         0x0110
#define NVE4_COMPUTE_INVALIDATE_CACHES 0x021c
#define NVE4_COMPUTE_SHARED_BASE       0x0214
#define NVE4_COMPUTE_LAUNCH_DESC_ADDR  0x02b4
#define NVE4_COMPUTE_LAUNCH            0x02bc
#define NVE4_COMPUTE_UNK0310           0x0310
#define NVE4_COMPUTE_LOCAL_BASE        0x077c
#define NVE4_COMPUTE_CODE_ADDRESS_HIGH 0x1608
#define NVE4_COMPUTE_CODE_ADDRESS_LOW  0x160c
#define NVE4_COMPUTE_FLUSH             0x1698

int main(void) {
    printf("============================================================\n");
    printf("  NVIDIA GeForce GT 750M (Kepler GK107 sm_30)\n");
    printf("  Live Mathematical Compute: output[tid] = tid + 42\n");
    printf("  Mesa-Verified 64-bit SASS + VRAM CB0 + Global DMA ST.E\n");
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
    printf("[+] DRM device wrapped OK (GK%02x Kepler)\n", dev->chipset);

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
    printf("[+] FIFO Channel GR allocated.\n");

    struct nouveau_object *compute_obj = NULL;
    ret = nouveau_object_new(chan, 0xbeef0001, KEPLER_COMPUTE_A, NULL, 0, &compute_obj);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_object_new(KEPLER_COMPUTE_A) failed: %d\n", ret);
        return 1;
    }
    printf("[+] Object 0xbeef0001 (KEPLER_COMPUTE_A) instantiated.\n");

    struct nouveau_pushbuf *push = NULL;
    ret = nouveau_pushbuf_new(client, chan, 4, 32 * 1024, true, &push);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_pushbuf_new failed: %d\n", ret);
        return 1;
    }

    /* 1. Output Buffer en GART (accesible por CPU y GPU) */
    struct nouveau_bo *output_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_GART, 256, 4096, NULL, &output_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate output_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(output_bo, NOUVEAU_BO_RDWR, client);
    volatile uint32_t *output_ptr = (volatile uint32_t *)output_bo->map;
    for (int i = 0; i < 64; i++) {
        output_ptr[i] = 0xDEADBEEF;
    }
    printf("[+] Output Buffer at GART GPU VA: 0x%016llx\n", (unsigned long long)output_bo->offset);

    /* 2. Constant Buffer 0 (CB0) en VRAM (requerido por el Constant Cache de Kepler) */
    struct nouveau_bo *cb0_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_VRAM, 256, 4096, NULL, &cb0_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate cb0_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(cb0_bo, NOUVEAU_BO_WR, client);
    memset(cb0_bo->map, 0, 4096);
    uint32_t *cb0_ptr = (uint32_t *)cb0_bo->map;

    /* blockDim / gridDim en CB0 según ABI oficial */
    cb0_ptr[0x08 / 4] = 32; /* blockDim.x */
    cb0_ptr[0x0c / 4] = 1;  /* blockDim.y */
    cb0_ptr[0x10 / 4] = 1;  /* blockDim.z */
    cb0_ptr[0x14 / 4] = 1;  /* gridDim.x   */
    cb0_ptr[0x18 / 4] = 1;  /* gridDim.y   */
    cb0_ptr[0x1c / 4] = 1;  /* gridDim.z   */

    /* Puntero de 64 bits a output_bo en CB0 offset 0x148 / 0x14c */
    cb0_ptr[0x148 / 4]     = (uint32_t)(output_bo->offset & 0xffffffff);
    cb0_ptr[(0x148 / 4) + 1] = (uint32_t)(output_bo->offset >> 32);

    printf("[+] CB0 at VRAM GPU VA:           0x%016llx\n", (unsigned long long)cb0_bo->offset);
    printf("    c0[0x148]: output_ptr = 0x%016llx\n", (unsigned long long)output_bo->offset);

    /* 3. SASS Code Buffer en VRAM */
    struct nouveau_bo *code_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_VRAM, 256, 4096, NULL, &code_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate code_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(code_bo, NOUVEAU_BO_WR, client);
    uint32_t *code = (uint32_t *)code_bo->map;
    memset(code, 0, 4096);

    /* === BUNDLE 0 (Bytes 0x00..0x3F): Math Computation + Global Store === */
    // [0] SW Sched Control Word para 7 instrucciones
    code[0] = 0x00000007 | (0x24 << 4) | (0x24 << 12) | (0x24 << 20) | (0x2eULL << 28);
    code[1] = 0x20000000 | (0x2e >> 4);

    // [1] S2R R0, SR_TID.X
    code[2] = 0x84001c04;
    code[3] = 0x2c000000;

    // [2] MOV R2, c[0x0][0x148] (Output VA Low)
    code[4] = 0x20009de4;
    code[5] = 0x28004005;

    // [3] MOV R3, c[0x0][0x14c] (Output VA High)
    code[6] = 0x3000dde4;
    code[7] = 0x28004005;

    // [4] SHL R4, R0, 2 (R4 = tid * 4 bytes)
    code[8] = 0x08011c03;
    code[9] = 0x6000c000;

    // [5] ADD R2, R2, R4 (R2 = base_lo + tid * 4)
    code[10] = 0x10209c03;
    code[11] = 0x48000000;

    // [6] ADD R5, R0, 42 (R5 = tid + 42)
    code[12] = 0xa8015c03;
    code[13] = 0x4800c000;

    // [7] ST.E [R2], R5 (Global Store: output[tid] = R5)
    code[14] = 0x00215c85;
    code[15] = 0x94000000;

    /* === BUNDLE 1 (Bytes 0x40..0x7F): EXIT + Warp Termination === */
    // [0] SW Sched Control Word
    code[16] = 0x000002e7;
    code[17] = 0x20000000;

    // [1] EXIT
    code[18] = 0x00001de7;
    code[19] = 0x80000000;

    printf("[+] SASS Microkernel uploaded to VRAM at: 0x%016llx (128 bytes)\n",
           (unsigned long long)code_bo->offset);

    /* 4. Query Buffer para Semáforo de Hardware en GART */
    struct nouveau_bo *query_bo = NULL;
    ret = nouveau_bo_new(dev, NOUVEAU_BO_GART, 256, 4096, NULL, &query_bo);
    if (ret != 0) {
        fprintf(stderr, "[-] Failed to allocate query_bo: %d\n", ret);
        return 1;
    }
    nouveau_bo_map(query_bo, NOUVEAU_BO_RDWR, client);
    volatile uint32_t *query_ptr = (volatile uint32_t *)query_bo->map;
    query_ptr[0] = 0xDEADBEEF;

    /* 5. Descriptor QMD v00_06 (256 bytes) */
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
    NVA0C0_QMDV00_06_VAL_SET(qmd, SASS_VERSION, 0x30);

    NVA0C0_QMDV00_06_VAL_SET(qmd, PROGRAM_OFFSET, 0);

    /* 1 Bloque x 32 Hilos (1 Warp) */
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_WIDTH, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_HEIGHT, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_DEPTH, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION0, 32);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION1, 1);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION2, 1);

    NVA0C0_QMDV00_06_DEF_SET(qmd, L1_CONFIGURATION, DIRECTLY_ADDRESSABLE_MEMORY_SIZE_16KB);
    NVA0C0_QMDV00_06_VAL_SET(qmd, REGISTER_COUNT, 16);
    NVA0C0_QMDV00_06_VAL_SET(qmd, BARRIER_COUNT, 0);

    /* CB0 mapping (VRAM) */
    NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_ADDR_LOWER, 0, (uint32_t)(cb0_bo->offset & 0xffffffff));
    NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_ADDR_UPPER, 0, (uint32_t)(cb0_bo->offset >> 32));
    NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_SIZE, 0, 4096);
    NVA0C0_QMDV00_06_DEF_SET(qmd, CONSTANT_BUFFER_VALID, 0, TRUE);

    /* Semáforo de hardware al terminar */
    NVA0C0_QMDV00_06_DEF_SET(qmd, SEMAPHORE_RELEASE_ENABLE0, TRUE);
    NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_ADDRESS_LOWER, (uint32_t)(query_bo->offset & 0xffffffff));
    NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_ADDRESS_UPPER, (uint32_t)(query_bo->offset >> 32));
    NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_PAYLOAD, 0xCAFEBABE);
    NVA0C0_QMDV00_06_DEF_SET(qmd, RELEASE0_STRUCTURE_SIZE, ONE_WORD);

    printf("[+] QMD v00_06 configured (REGISTER_COUNT=16, CB0=4096B VRAM, WARP=32).\n");

    /* 6. Pushbuffer: registrar los 5 BOs en MMU */
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

    /* 7. Emitir métodos al pushbuffer */
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

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_INVALIDATE_CACHES, 1);
    *push->cur++ = 0x00001011;

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_FLUSH, 1);
    *push->cur++ = 0x00000001;

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LAUNCH_DESC_ADDR, 1);
    *push->cur++ = (uint32_t)(desc_bo->offset >> 8);

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LAUNCH, 1);
    *push->cur++ = 0x00000003;

    *push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_SERIALIZE, 1);
    *push->cur++ = 0x00000000;

    printf("[*] Kicking compute pushbuffer to hardware...\n");
    ret = nouveau_pushbuf_kick(push, chan);
    if (ret != 0) {
        fprintf(stderr, "[-] nouveau_pushbuf_kick failed: %d\n", ret);
        return 1;
    }

    /* 8. Esperar finalización del semáforo */
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
        printf("    Thread %02d: GPU wrote %3d (0x%08x) | Expected %3d --> %s\n",
               i, (int)val, val, expected, (val == expected) ? "PASS" : "FAIL");
        if (val != expected) {
            errors++;
        }
    }

    printf("------------------------------------------------------------\n");
    if (query_ptr[0] == 0xCAFEBABE && errors == 0) {
        printf("  >>> [SUCCESS] 32/32 THREADS PRODUCED 100%% EXACT MATH! <<<\n");
        printf("  NVIDIA GeForce GT 750M (GK107) computed in real silicon!\n");
        printf("  Constant Buffer 0 + SASS ST.E + Global DMA: 100%% VERIFIED.\n");
    } else {
        printf("  [NOTICE] Errors: %d / 32 threads.\n", errors);
    }
    printf("============================================================\n");

    /* Cleanup */
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
