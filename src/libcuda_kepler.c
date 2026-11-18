#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <nouveau/nouveau.h>

#include <elf.h>

#include "cuda.h"
#include "gpu_turbo_governor.h"
#include "drf.h"
#include "cla0c0qmd.h"

#define KEPLER_COMPUTE_A           0xa0c0
#define MAX_ALLOCS                1024

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

struct alloc_entry {
    CUdeviceptr dptr;
    struct nouveau_bo *bo;
    size_t size;
    int in_use;
    int is_module_owned;
};

#define NVINFO_EIFMT_BVAL 2
#define NVINFO_EIFMT_HVAL 3
#define NVINFO_EIFMT_SVAL 4

struct nvinfo_attribute_header_t {
    uint8_t format;
    uint8_t attribute;
    union {
        struct { uint8_t value; } bval;
        struct { uint16_t value; } hval;
        struct { uint16_t size; } sval;
    };
};

#define NVINFO_EIATTR_PARAM_CBANK 10
struct nvinfo_param_cbank_t {
    uint32_t unknown1;
    uint16_t params_offset;
    uint16_t params_size;
};

#define NVINFO_EIATTR_KPARAM_INFO 23
struct nvinfo_kparam_info_t {
    uint32_t index;
    uint16_t ordinal;
    uint16_t offset;
    uint32_t packed;
};

#define NVINFO_EIATTR_MIN_STACK_SIZE 18
struct nvinfo_attr_min_stack_size_t {
    uint32_t symbol_index;
    uint32_t size;
};

#define NVINFO_EIATTR_REGCOUNT 47
struct nvinfo_attr_regcount_t {
    uint32_t symbol_index;
    uint32_t count;
};

#define NVINFO_EIATTR_NUM_BARRIERS 76

struct kernel_param {
    uint16_t ordinal;
    uint16_t offset;
    uint16_t size;
};

struct CUfunc_st;

struct CUmod_st {
    struct nouveau_bo *code_bo;
    size_t code_size;
    void *elf_data;
    size_t elf_size;
    int is_elf;
    struct CUfunc_st *funcs;
    struct nouveau_bo *cb_bo[8];
    size_t cb_size[8];
};

struct CUfunc_st {
    struct CUmod_st *mod;
    struct nouveau_bo *code_bo;
    struct nouveau_bo *tls_bo;
    uint64_t code_offset;
    uint32_t num_regs;
    uint32_t shared_size;
    uint32_t local_size;
    uint32_t tls_mp_size;
    uint32_t num_barriers;
    uint32_t param_base_offset;
    int num_params;
    struct kernel_param params[64];
    struct CUfunc_st *next;
};

struct CUctx_st {
    struct nouveau_device *dev;
    struct nouveau_client *client;
    struct nouveau_object *chan;
    struct nouveau_pushbuf *push;
    struct nouveau_object *compute_obj;
    struct nouveau_bo *fence_bo;
    struct nouveau_bo *desc_bo;
    struct nouveau_bo *cb0_bo;
    struct alloc_entry allocs[MAX_ALLOCS];
    uint32_t launch_seq;
    pthread_mutex_t mutex;
};

struct CUstream_st {
    struct CUctx_st *ctx;
    unsigned int flags;
    int priority;
    uint32_t last_seq;
};

struct CUevent_st {
    struct CUctx_st *ctx;
    uint32_t seq;
    uint64_t timestamp_ns;
    int recorded;
    unsigned int flags;
};

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static struct nouveau_device *g_dev = NULL;
static int g_fd = -1;
static __thread struct CUctx_st *g_current_ctx = NULL;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

CUresult cuInit(unsigned int Flags) {
    (void)Flags;
    pthread_mutex_lock(&g_init_mutex);
    if (g_dev != NULL) {
        pthread_mutex_unlock(&g_init_mutex);
        return CUDA_SUCCESS;
    }

    const char *nodes[] = { "/dev/dri/renderD128", "/dev/dri/card0", NULL };
    for (int i = 0; nodes[i] != NULL; i++) {
        g_fd = open(nodes[i], O_RDWR);
        if (g_fd >= 0) break;
    }

    if (g_fd < 0) {
        pthread_mutex_unlock(&g_init_mutex);
        return CUDA_ERROR_NO_DEVICE;
    }

    int ret = nouveau_device_wrap(g_fd, 0, &g_dev);
    if (ret != 0 || !g_dev) {
        close(g_fd);
        g_fd = -1;
        pthread_mutex_unlock(&g_init_mutex);
        return CUDA_ERROR_UNKNOWN;
    }

    gpu_governor_client_init();

    pthread_mutex_unlock(&g_init_mutex);
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *driverVersion) {
    if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
    *driverVersion = 10020; // Canonical CUDA 10.2 for sm_30
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    if (!g_dev) {
        CUresult r = cuInit(0);
        if (r != CUDA_SUCCESS) return r;
    }
    *count = 1;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (!device || ordinal != 0) return CUDA_ERROR_INVALID_DEVICE;
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (!name || len <= 0 || dev != 0) return CUDA_ERROR_INVALID_VALUE;
    snprintf(name, len, "NVIDIA GeForce GT 750M [Kepler GK107 sm_30 Native]");
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    if (!bytes || dev != 0 || !g_dev) return CUDA_ERROR_INVALID_VALUE;
    *bytes = (size_t)g_dev->vram_size;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    if (!pi || dev != 0) return CUDA_ERROR_INVALID_VALUE;
    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            *pi = 2; // GT 750M (2 SMX, 384 CUDA cores)
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
            *pi = 3;
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
            *pi = 32;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *pi = 1024;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
            *pi = 1024;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
            *pi = 1024;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
            *pi = 64;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
            *pi = 2147483647;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
            *pi = 65535;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
            *pi = 65535;
            break;
        case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
            *pi = 926000; // Turbo boost peak kHz
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
            *pi = 49152; // 48 KiB
            break;
        case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
            *pi = 65536; // 64 KiB
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
            *pi = 65536; // 64K registers per SMX
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
            *pi = 65536;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
            *pi = 2048; // Max 2048 active threads per SMX on Kepler
            break;
        case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
            *pi = 262144; // 256 KiB L2 cache on GK107
            break;
        case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
            *pi = 2500000; // 2.5 GHz GDDR5
            break;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
            *pi = 128; // 128-bit memory bus
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
            *pi = 2147483647;
            break;
        case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
            *pi = 512;
            break;
        case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_INTEGRATED:
            *pi = 0; // Discrete dGPU
            break;
        case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
            *pi = 0; // Default
            break;
        default:
            *pi = 0; // Capacidades no soportadas o extendidas devuelven 0
            break;
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    (void)flags;
    if (!pctx || dev != 0 || !g_dev) return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = (struct CUctx_st *)calloc(1, sizeof(struct CUctx_st));
    if (!ctx) return CUDA_ERROR_OUT_OF_MEMORY;

    ctx->dev = g_dev;
    pthread_mutex_init(&ctx->mutex, NULL);

    int ret = nouveau_client_new(g_dev, &ctx->client);
    if (ret != 0) {
        free(ctx);
        return CUDA_ERROR_UNKNOWN;
    }

    struct nve0_fifo fifo_args = { .engine = NVE0_FIFO_ENGINE_GR };
    ret = nouveau_object_new(&g_dev->object, 0, NOUVEAU_FIFO_CHANNEL_CLASS,
                             &fifo_args, sizeof(fifo_args), &ctx->chan);
    if (ret != 0) {
        nouveau_client_del(&ctx->client);
        free(ctx);
        return CUDA_ERROR_UNKNOWN;
    }

    ret = nouveau_object_new(ctx->chan, 0xbeef0001, KEPLER_COMPUTE_A, NULL, 0, &ctx->compute_obj);
    if (ret != 0) {
        nouveau_object_del(&ctx->chan);
        nouveau_client_del(&ctx->client);
        free(ctx);
        return CUDA_ERROR_UNKNOWN;
    }

    ret = nouveau_pushbuf_new(ctx->client, ctx->chan, 4, 32 * 1024, true, &ctx->push);
    if (ret != 0) {
        nouveau_object_del(&ctx->compute_obj);
        nouveau_object_del(&ctx->chan);
        nouveau_client_del(&ctx->client);
        free(ctx);
        return CUDA_ERROR_UNKNOWN;
    }

    // 1. Fence de hardware persistente en GART para sincronización
    ret = nouveau_bo_new(ctx->dev, NOUVEAU_BO_GART, 256, 4096, NULL, &ctx->fence_bo);
    if (ret == 0 && ctx->fence_bo) {
        nouveau_bo_map(ctx->fence_bo, NOUVEAU_BO_RDWR, ctx->client);
        *(volatile uint32_t *)ctx->fence_bo->map = 0;
    }

    // 2. QMD v00_06 descriptor en VRAM (256 bytes)
    nouveau_bo_new(ctx->dev, NOUVEAU_BO_VRAM, 256, 256, NULL, &ctx->desc_bo);

    // 3. Constant Buffer 0 (CB0) en VRAM (4096 bytes)
    nouveau_bo_new(ctx->dev, NOUVEAU_BO_VRAM, 256, 4096, NULL, &ctx->cb0_bo);

    ctx->launch_seq = 0;
    *pctx = ctx;
    g_current_ctx = ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    pthread_mutex_lock(&ctx->mutex);

    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (ctx->allocs[i].in_use && ctx->allocs[i].bo) {
            nouveau_bo_ref(NULL, &ctx->allocs[i].bo);
            ctx->allocs[i].in_use = 0;
        }
    }

    if (ctx->cb0_bo) nouveau_bo_ref(NULL, &ctx->cb0_bo);
    if (ctx->desc_bo) nouveau_bo_ref(NULL, &ctx->desc_bo);
    if (ctx->fence_bo) nouveau_bo_ref(NULL, &ctx->fence_bo);
    if (ctx->push) nouveau_pushbuf_del(&ctx->push);
    if (ctx->compute_obj) nouveau_object_del(&ctx->compute_obj);
    if (ctx->chan) nouveau_object_del(&ctx->chan);
    if (ctx->client) nouveau_client_del(&ctx->client);

    pthread_mutex_unlock(&ctx->mutex);
    pthread_mutex_destroy(&ctx->mutex);

    if (g_current_ctx == ctx) g_current_ctx = NULL;
    free(ctx);

    gpu_governor_client_close();
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = g_current_ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    g_current_ctx = ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    if (!g_current_ctx) return CUDA_ERROR_INVALID_CONTEXT;
    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);

    int kick_err = 0;
    if (ctx->push && ctx->chan) {
        kick_err = nouveau_pushbuf_kick(ctx->push, ctx->chan);
    }

    int wait_err = 0;
    if (ctx->fence_bo && ctx->fence_bo->map) {
        volatile uint32_t *fence_ptr = (volatile uint32_t *)ctx->fence_bo->map;
        uint32_t target_seq = ctx->launch_seq;
        // Espera con deadline monotónico real (5000 ms = 5.0 s)
        struct timespec start_ts, now_ts;
        clock_gettime(CLOCK_MONOTONIC, &start_ts);
        uint64_t deadline_ns = (uint64_t)start_ts.tv_sec * 1000000000ULL + start_ts.tv_nsec + 5000000000ULL;
        while ((int32_t)(*fence_ptr - target_seq) < 0) {
            clock_gettime(CLOCK_MONOTONIC, &now_ts);
            uint64_t now_ns = (uint64_t)now_ts.tv_sec * 1000000000ULL + now_ts.tv_nsec;
            if (now_ns >= deadline_ns) {
                // Timeout alcanzado: sincronización forzada por kernel DRM
                wait_err = nouveau_bo_wait(ctx->fence_bo, NOUVEAU_BO_RDWR, ctx->client);
                break;
            }
            usleep(50);
        }
    }
    pthread_mutex_unlock(&ctx->mutex);

    gpu_governor_release_turbo();

    if (kick_err != 0 || wait_err != 0 || (ctx->fence_bo && (int32_t)(*(volatile uint32_t *)ctx->fence_bo->map - ctx->launch_seq) < 0)) {
        return CUDA_ERROR_LAUNCH_TIMEOUT;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    if (!dptr || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    if (!g_current_ctx) return CUDA_ERROR_INVALID_CONTEXT;

    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);

    int slot = -1;
    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (!ctx->allocs[i].in_use) {
            slot = i;
            break;
        }
    }
    if (slot == -1) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    struct nouveau_bo *bo = NULL;
    // Asignar en VRAM con alineación natural de página
    int ret = nouveau_bo_new(ctx->dev, NOUVEAU_BO_VRAM, 256, bytesize, NULL, &bo);
    if (ret != 0 || !bo) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    // La dirección del dispositivo es la dirección virtual real (GPU VA) del canal
    CUdeviceptr dev_addr = (CUdeviceptr)bo->offset;

    ctx->allocs[slot].dptr = dev_addr;
    ctx->allocs[slot].bo = bo;
    ctx->allocs[slot].size = bytesize;
    ctx->allocs[slot].is_module_owned = 0;
    ctx->allocs[slot].in_use = 1;

    *dptr = dev_addr;
    pthread_mutex_unlock(&ctx->mutex);
    return CUDA_SUCCESS;
}

static struct alloc_entry *find_alloc(struct CUctx_st *ctx, CUdeviceptr dptr) {
    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (ctx->allocs[i].in_use &&
            dptr >= ctx->allocs[i].dptr &&
            dptr < ctx->allocs[i].dptr + ctx->allocs[i].size) {
            return &ctx->allocs[i];
        }
    }
    return NULL;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    if (!g_current_ctx) return CUDA_ERROR_INVALID_CONTEXT;
    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);

    // Liberación estricta por dirección base y que no sea propiedad del módulo
    struct alloc_entry *entry = NULL;
    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (ctx->allocs[i].in_use && ctx->allocs[i].dptr == dptr) {
            entry = &ctx->allocs[i];
            break;
        }
    }
    if (!entry || entry->is_module_owned) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    nouveau_bo_ref(NULL, &entry->bo);
    entry->in_use = 0;
    entry->bo = NULL;
    entry->dptr = 0;
    entry->size = 0;
    entry->is_module_owned = 0;

    pthread_mutex_unlock(&ctx->mutex);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    if (!srcHost || !g_current_ctx) return CUDA_ERROR_INVALID_VALUE;
    if (ByteCount == 0) return CUDA_SUCCESS;
    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);

    struct alloc_entry *entry = find_alloc(ctx, dstDevice);
    if (!entry) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint64_t offset = dstDevice - entry->dptr;
    if (offset > entry->size || ByteCount > entry->size - offset) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    int ret = nouveau_bo_map(entry->bo, NOUVEAU_BO_WR, ctx->client);
    if (ret != 0 || !entry->bo->map) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_MAP_FAILED;
    }

    memcpy((uint8_t *)entry->bo->map + offset, srcHost, ByteCount);
    __builtin_ia32_sfence();

    pthread_mutex_unlock(&ctx->mutex);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!dstHost || !g_current_ctx) return CUDA_ERROR_INVALID_VALUE;
    if (ByteCount == 0) return CUDA_SUCCESS;
    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);

    struct alloc_entry *entry = find_alloc(ctx, srcDevice);
    if (!entry) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint64_t offset = srcDevice - entry->dptr;
    if (offset > entry->size || ByteCount > entry->size - offset) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    int ret = nouveau_bo_map(entry->bo, NOUVEAU_BO_RD, ctx->client);
    if (ret != 0 || !entry->bo->map) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_MAP_FAILED;
    }

    memcpy(dstHost, (const uint8_t *)entry->bo->map + offset, ByteCount);
    pthread_mutex_unlock(&ctx->mutex);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    if (!g_current_ctx) return CUDA_ERROR_INVALID_CONTEXT;
    if (N == 0) return CUDA_SUCCESS;
    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);

    struct alloc_entry *entry = find_alloc(ctx, dstDevice);
    if (!entry) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint64_t offset = dstDevice - entry->dptr;
    if (offset > entry->size || N > entry->size - offset) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    int ret = nouveau_bo_map(entry->bo, NOUVEAU_BO_WR, ctx->client);
    if (ret != 0 || !entry->bo->map) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_MAP_FAILED;
    }

    memset((uint8_t *)entry->bo->map + offset, uc, N);
    __builtin_ia32_sfence();

    pthread_mutex_unlock(&ctx->mutex);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    if (!g_current_ctx) return CUDA_ERROR_INVALID_CONTEXT;
    if (N == 0) return CUDA_SUCCESS;
    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);

    struct alloc_entry *entry = find_alloc(ctx, dstDevice);
    if (!entry) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint64_t offset = dstDevice - entry->dptr;
    if (offset > entry->size || (entry->size - offset) / sizeof(uint32_t) < N) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    int ret = nouveau_bo_map(entry->bo, NOUVEAU_BO_WR, ctx->client);
    if (ret != 0 || !entry->bo->map) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_MAP_FAILED;
    }

    uint32_t *p = (uint32_t *)((uint8_t *)entry->bo->map + offset);
    for (size_t i = 0; i < N; i++) p[i] = ui;
    __builtin_ia32_sfence();

    pthread_mutex_unlock(&ctx->mutex);
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    if (!free || !total) return CUDA_ERROR_INVALID_VALUE;
    if (!g_dev) return CUDA_ERROR_INVALID_CONTEXT;
    *total = (size_t)g_dev->vram_size;
    size_t used = 0;
    if (g_current_ctx) {
        struct CUctx_st *ctx = g_current_ctx;
        pthread_mutex_lock(&ctx->mutex);
        for (int i = 0; i < MAX_ALLOCS; i++) {
            if (ctx->allocs[i].in_use) {
                used += ctx->allocs[i].size;
            }
        }
        if (ctx->cb0_bo) used += ctx->cb0_bo->size;
        if (ctx->desc_bo) used += ctx->desc_bo->size;
        pthread_mutex_unlock(&ctx->mutex);
    }
    if (used > *total) *free = 0;
    else *free = *total - used;
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo(size_t *free, size_t *total) {
    return cuMemGetInfo_v2(free, total);
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    (void)ElementSizeBytes;
    if (!dptr || !pPitch || WidthInBytes == 0 || Height == 0) return CUDA_ERROR_INVALID_VALUE;
    size_t pitch = (WidthInBytes + 255) & ~255; // Alineación 256 bytes Kepler
    if (pitch < WidthInBytes) return CUDA_ERROR_OUT_OF_MEMORY;
    size_t total_size = pitch * Height;
    if (Height != 0 && total_size / Height != pitch) return CUDA_ERROR_OUT_OF_MEMORY;
    CUresult r = cuMemAlloc(dptr, total_size);
    if (r == CUDA_SUCCESS) {
        *pPitch = pitch;
    }
    return r;
}

CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    if (!pbase || !psize || !g_current_ctx) return CUDA_ERROR_INVALID_VALUE;
    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);
    struct alloc_entry *entry = find_alloc(ctx, dptr);
    if (!entry) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_NOT_FOUND;
    }
    *pbase = entry->dptr;
    *psize = entry->size;
    pthread_mutex_unlock(&ctx->mutex);
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    if (!data || !g_current_ctx) return CUDA_ERROR_INVALID_VALUE;
    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);
    struct alloc_entry *entry = find_alloc(ctx, ptr);
    pthread_mutex_unlock(&ctx->mutex);
    if (!entry) return CUDA_ERROR_INVALID_VALUE;

    switch (attribute) {
        case CU_POINTER_ATTRIBUTE_CONTEXT:
            *(CUcontext *)data = ctx;
            break;
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
            *(CUmemorytype *)data = CU_MEMORYTYPE_DEVICE;
            break;
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
            *(CUdeviceptr *)data = ptr;
            break;
        case CU_POINTER_ATTRIBUTE_HOST_POINTER:
            *(void **)data = NULL;
            break;
        case CU_POINTER_ATTRIBUTE_IS_MANAGED:
            *(int *)data = 0;
            break;
        case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
            *(int *)data = 0;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr) {
    if (!attributes || !data) return CUDA_ERROR_INVALID_VALUE;
    for (unsigned int i = 0; i < numAttributes; i++) {
        CUresult r = cuPointerGetAttribute(data[i], attributes[i], ptr);
        if (r != CUDA_SUCCESS) return r;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!g_current_ctx) return CUDA_ERROR_INVALID_CONTEXT;
    if (ByteCount == 0) return CUDA_SUCCESS;
    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);

    struct alloc_entry *dst_entry = find_alloc(ctx, dstDevice);
    struct alloc_entry *src_entry = find_alloc(ctx, srcDevice);
    if (!dst_entry || !src_entry) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint64_t dst_off = dstDevice - dst_entry->dptr;
    uint64_t src_off = srcDevice - src_entry->dptr;
    if (dst_off > dst_entry->size || ByteCount > dst_entry->size - dst_off ||
        src_off > src_entry->size || ByteCount > src_entry->size - src_off) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_INVALID_VALUE;
    }

    int r1 = nouveau_bo_map(src_entry->bo, NOUVEAU_BO_RD, ctx->client);
    int r2 = nouveau_bo_map(dst_entry->bo, NOUVEAU_BO_WR, ctx->client);
    if (r1 != 0 || r2 != 0 || !src_entry->bo->map || !dst_entry->bo->map) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_MAP_FAILED;
    }

    memcpy((uint8_t *)dst_entry->bo->map + dst_off,
           (const uint8_t *)src_entry->bo->map + src_off,
           ByteCount);
    __builtin_ia32_sfence();

    pthread_mutex_unlock(&ctx->mutex);
    return CUDA_SUCCESS;
}

// Símbolos canónicos _v2 para compatibilidad con software compilado para CUDA oficial
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) { return cuCtxCreate(pctx, flags, dev); }
CUresult cuCtxDestroy_v2(CUcontext ctx) { return cuCtxDestroy(ctx); }
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) { return cuMemAlloc(dptr, bytesize); }
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) { return cuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes); }
CUresult cuMemFree_v2(CUdeviceptr dptr) { return cuMemFree(dptr); }
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) { return cuMemcpyHtoD(dstDevice, srcHost, ByteCount); }
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) { return cuMemcpyDtoH(dstHost, srcDevice, ByteCount); }
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) { return cuMemcpyDtoD(dstDevice, srcDevice, ByteCount); }

/* === GESTIÓN DE MÓDULOS Y KERNELS SASS === */

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!module || !image || !g_current_ctx) return CUDA_ERROR_INVALID_VALUE;
    struct CUctx_st *ctx = g_current_ctx;
    pthread_mutex_lock(&ctx->mutex);

    struct CUmod_st *mod = (struct CUmod_st *)calloc(1, sizeof(struct CUmod_st));
    if (!mod) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    // Reservar 4096 bytes en VRAM para alojar el código SASS alineado a 256 B
    mod->code_size = 4096;
    int ret = nouveau_bo_new(ctx->dev, NOUVEAU_BO_VRAM, 256, mod->code_size, NULL, &mod->code_bo);
    if (ret != 0 || !mod->code_bo) {
        free(mod);
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    const uint8_t *magic = (const uint8_t *)image;
    if (magic[0] == 0x7f && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F') {
        // Formato ELF nativo (cubin generado por ptxas / nvcc oficial)
        Elf64_Ehdr *ehdr = (Elf64_Ehdr *)image;
        Elf64_Shdr *shdrs = (Elf64_Shdr *)((uint8_t *)image + ehdr->e_shoff);
        size_t max_off = ehdr->e_shoff + ehdr->e_shnum * ehdr->e_shentsize;
        for (int i = 0; i < ehdr->e_shnum; i++) {
            size_t end = shdrs[i].sh_offset + shdrs[i].sh_size;
            if (end > max_off) max_off = end;
        }

        mod->elf_data = malloc(max_off);
        if (mod->elf_data) {
            memcpy(mod->elf_data, image, max_off);
            mod->elf_size = max_off;
            mod->is_elf = 1;

            // [PASO 3] Escanear y mapear bancos de memoria constante de usuario .nv.constant<N>
            const char *shstrtab = (const char *)mod->elf_data + shdrs[ehdr->e_shstrndx].sh_offset;
            for (int i = 0; i < ehdr->e_shnum; i++) {
                const char *sec_name = shstrtab + shdrs[i].sh_name;
                int bank = -1;
                if (sscanf(sec_name, ".nv.constant%d", &bank) == 1 && bank >= 1 && bank < 8) {
                    size_t bo_sz = (shdrs[i].sh_size + 0xff) & ~0xff;
                    if (bo_sz < 256) bo_sz = 256;
                    int r = nouveau_bo_new(ctx->dev, NOUVEAU_BO_VRAM, 256, bo_sz, NULL, &mod->cb_bo[bank]);
                    if (r == 0 && mod->cb_bo[bank]) {
                        nouveau_bo_map(mod->cb_bo[bank], NOUVEAU_BO_WR, ctx->client);
                        memset(mod->cb_bo[bank]->map, 0, bo_sz);
                        if (shdrs[i].sh_size > 0) {
                            memcpy(mod->cb_bo[bank]->map, (uint8_t *)mod->elf_data + shdrs[i].sh_offset, shdrs[i].sh_size);
                            __builtin_ia32_sfence();
                        }
                        mod->cb_size[bank] = shdrs[i].sh_size;
                        // Registrar en ctx->allocs para permitir cuMemcpyHtoD directo
                        for (int a = 0; a < MAX_ALLOCS; a++) {
                            if (!ctx->allocs[a].in_use) {
                                ctx->allocs[a].dptr = (CUdeviceptr)mod->cb_bo[bank]->offset;
                                ctx->allocs[a].bo = mod->cb_bo[bank];
                                ctx->allocs[a].size = bo_sz;
                                ctx->allocs[a].is_module_owned = 1;
                                ctx->allocs[a].in_use = 1;
                                break;
                            }
                        }
                        printf("[DEBUG CONSTANT] Allocated CB%d for section '%s' (GPU VA=0x%lx, size=%zu)\n",
                               bank, sec_name, (unsigned long)mod->cb_bo[bank]->offset, shdrs[i].sh_size);
                    }
                }
            }
        }
    } else {
        // Binario plano SASS directo
        mod->is_elf = 0;
        nouveau_bo_map(mod->code_bo, NOUVEAU_BO_WR, ctx->client);
        memset(mod->code_bo->map, 0, mod->code_size);
        memcpy(mod->code_bo->map, image, 1024);
    }

    *module = mod;
    pthread_mutex_unlock(&ctx->mutex);
    return CUDA_SUCCESS;
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    if (!module || !fname) return CUDA_ERROR_INVALID_VALUE;
    FILE *f = fopen(fname, "rb");
    if (!f) return CUDA_ERROR_FILE_NOT_FOUND;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0 || size > 1024 * 1024) {
        fclose(f);
        return CUDA_ERROR_INVALID_IMAGE;
    }

    void *buf = malloc(size);
    if (!buf) {
        fclose(f);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    fread(buf, 1, size, f);
    fclose(f);

    CUresult r = cuModuleLoadData(module, buf);
    free(buf);
    return r;
}

CUresult cuFuncDestroy(CUfunction f) {
    if (!f) return CUDA_ERROR_INVALID_HANDLE;
    if (f->mod) {
        struct CUfunc_st **pp = &f->mod->funcs;
        while (*pp) {
            if (*pp == f) {
                *pp = f->next;
                break;
            }
            pp = &(*pp)->next;
        }
    }
    if (f->code_bo) nouveau_bo_ref(NULL, &f->code_bo);
    if (f->tls_bo)  nouveau_bo_ref(NULL, &f->tls_bo);
    free(f);
    return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule hmod) {
    if (!hmod) return CUDA_ERROR_INVALID_VALUE;
    struct CUfunc_st *f = hmod->funcs;
    while (f) {
        struct CUfunc_st *next = f->next;
        if (f->code_bo) nouveau_bo_ref(NULL, &f->code_bo);
        if (f->tls_bo)  nouveau_bo_ref(NULL, &f->tls_bo);
        free(f);
        f = next;
    }
    hmod->funcs = NULL;
    if (g_current_ctx) {
        for (int b = 0; b < 8; b++) {
            if (hmod->cb_bo[b]) {
                for (int a = 0; a < MAX_ALLOCS; a++) {
                    if (g_current_ctx->allocs[a].bo == hmod->cb_bo[b]) {
                        g_current_ctx->allocs[a].in_use = 0;
                        g_current_ctx->allocs[a].bo = NULL;
                        g_current_ctx->allocs[a].dptr = 0;
                        g_current_ctx->allocs[a].size = 0;
                        g_current_ctx->allocs[a].is_module_owned = 0;
                        break;
                    }
                }
                nouveau_bo_ref(NULL, &hmod->cb_bo[b]);
            }
        }
    }
    if (hmod->code_bo) nouveau_bo_ref(NULL, &hmod->code_bo);
    if (hmod->elf_data) free(hmod->elf_data);
    free(hmod);
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    if (!hfunc || !hmod || !g_current_ctx) return CUDA_ERROR_INVALID_VALUE;

    struct CUfunc_st *f = (struct CUfunc_st *)calloc(1, sizeof(struct CUfunc_st));
    if (!f) return CUDA_ERROR_OUT_OF_MEMORY;

    f->mod = hmod;
    f->num_regs = 16;
    f->param_base_offset = 0x140;

    if (hmod->is_elf && hmod->elf_data) {
        Elf64_Ehdr *ehdr = (Elf64_Ehdr *)hmod->elf_data;
        Elf64_Shdr *shdrs = (Elf64_Shdr *)((uint8_t *)hmod->elf_data + ehdr->e_shoff);
        const char *shstrtab = (const char *)hmod->elf_data + shdrs[ehdr->e_shstrndx].sh_offset;

        // 1. Buscar en tabla de símbolos .symtab
        int target_shndx = -1;
        int target_sym_idx = -1;
        const char *kernel_func_name = name;
        for (int i = 0; i < ehdr->e_shnum; i++) {
            if (shdrs[i].sh_type == SHT_SYMTAB) {
                Elf64_Shdr *symtab = &shdrs[i];
                const char *strtab = (const char *)hmod->elf_data + shdrs[symtab->sh_link].sh_offset;
                Elf64_Sym *syms = (Elf64_Sym *)((uint8_t *)hmod->elf_data + symtab->sh_offset);
                int num_syms = symtab->sh_size / sizeof(Elf64_Sym);

                for (int s = 0; s < num_syms; s++) {
                    const char *sname = strtab + syms[s].st_name;
                    if (name && strcmp(sname, name) == 0) {
                        target_shndx = syms[s].st_shndx;
                        target_sym_idx = s;
                        kernel_func_name = sname;
                        break;
                    }
                    if (!name && (ELF64_ST_TYPE(syms[s].st_info) == STT_FUNC)) {
                        target_shndx = syms[s].st_shndx;
                        target_sym_idx = s;
                        kernel_func_name = sname;
                        break;
                    }
                }
                break;
            }
        }

        // 2. Si no se halló por símbolo exacto, buscar sección .text.<name>
        if (target_shndx < 0) {
            for (int i = 0; i < ehdr->e_shnum; i++) {
                const char *sec_name = shstrtab + shdrs[i].sh_name;
                if ((shdrs[i].sh_flags & SHF_EXECINSTR) || strstr(sec_name, ".text")) {
                    if (!name || strstr(sec_name, name)) {
                        target_shndx = i;
                        if (!kernel_func_name) kernel_func_name = sec_name;
                        break;
                    }
                }
            }
        }

        if (target_shndx >= 0 && target_shndx < ehdr->e_shnum) {
            Elf64_Shdr *sec = &shdrs[target_shndx];
            const char *sec_name = shstrtab + sec->sh_name;

            // [PASO 2] Asignar un BO propio e independiente por función (aislamiento total en VRAM)
            size_t bo_size = (sec->sh_size + 0xff) & ~0xff;
            if (bo_size < 256) bo_size = 256;
            int ret = nouveau_bo_new(g_current_ctx->dev, NOUVEAU_BO_VRAM, 256, bo_size, NULL, &f->code_bo);
            if (ret != 0 || !f->code_bo) {
                free(f);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }

            nouveau_bo_map(f->code_bo, NOUVEAU_BO_WR, g_current_ctx->client);
            memset(f->code_bo->map, 0, bo_size);
            memcpy(f->code_bo->map, (uint8_t *)hmod->elf_data + sec->sh_offset, sec->sh_size);
            f->code_offset = 0;
            // En Kepler / Maxwell (ABI < 0x41), el conteo de barreras de control reside en (sh_flags >> 20) & 0xf
            f->num_barriers = (sec->sh_flags >> 20) & 0xf;
            if ((sec->sh_info >> 24) > 0) {
                f->num_regs = (sec->sh_info >> 24);
            }

            // [PASO 1 & 3] Parsear .nv.info y .nv.info.<func> para metadata exacta de parámetros y QMD
            char info_sec_name[512];
            snprintf(info_sec_name, sizeof(info_sec_name), ".nv.info.%s", sec_name);
            const char *bare_name = sec_name;
            if (strncmp(bare_name, ".text.", 6) == 0) bare_name += 6;
            char info_sec_name2[512];
            snprintf(info_sec_name2, sizeof(info_sec_name2), ".nv.info.%s", bare_name);

            char shared_sec_name[512];
            snprintf(shared_sec_name, sizeof(shared_sec_name), ".nv.shared.%s", bare_name);

            char local_sec_name[512];
            snprintf(local_sec_name, sizeof(local_sec_name), ".nv.local.%s", bare_name);

            for (int i = 0; i < ehdr->e_shnum; i++) {
                const char *sname = shstrtab + shdrs[i].sh_name;
                // Memoria compartida estática del kernel
                if (strcmp(sname, shared_sec_name) == 0) {
                    f->shared_size = shdrs[i].sh_size;
                }
                // Memoria local (spills/stack) declarada en sección
                if (strcmp(sname, local_sec_name) == 0) {
                    if (shdrs[i].sh_size > f->local_size) f->local_size = shdrs[i].sh_size;
                }
                // Registro global de registros y stack frame
                if (strcmp(sname, ".nv.info") == 0) {
                    const uint8_t *ptr = (const uint8_t *)hmod->elf_data + shdrs[i].sh_offset;
                    const uint8_t *end = ptr + shdrs[i].sh_size;
                    while (ptr + sizeof(struct nvinfo_attribute_header_t) <= end) {
                        const struct nvinfo_attribute_header_t *hdr = (const struct nvinfo_attribute_header_t *)ptr;
                        ptr += sizeof(struct nvinfo_attribute_header_t);
                        if (hdr->attribute == NVINFO_EIATTR_REGCOUNT && hdr->format == NVINFO_EIFMT_SVAL) {
                            const struct nvinfo_attr_regcount_t *attr = (const struct nvinfo_attr_regcount_t *)ptr;
                            if (target_sym_idx < 0 || attr->symbol_index == (uint32_t)target_sym_idx) {
                                f->num_regs = attr->count;
                            }
                        } else if (hdr->attribute == NVINFO_EIATTR_MIN_STACK_SIZE && hdr->format == NVINFO_EIFMT_SVAL) {
                            const struct nvinfo_attr_min_stack_size_t *attr = (const struct nvinfo_attr_min_stack_size_t *)ptr;
                            if (target_sym_idx < 0 || attr->symbol_index == (uint32_t)target_sym_idx) {
                                if (attr->size > f->local_size) f->local_size = attr->size;
                            }
                        }
                        if (hdr->format == NVINFO_EIFMT_SVAL) ptr += hdr->sval.size;
                    }
                }
                // Registro específico del kernel (parámetros y recursos)
                if (strcmp(sname, info_sec_name) == 0 || strcmp(sname, info_sec_name2) == 0) {
                    const uint8_t *ptr = (const uint8_t *)hmod->elf_data + shdrs[i].sh_offset;
                    const uint8_t *end = ptr + shdrs[i].sh_size;
                    while (ptr + sizeof(struct nvinfo_attribute_header_t) <= end) {
                        const struct nvinfo_attribute_header_t *hdr = (const struct nvinfo_attribute_header_t *)ptr;
                        ptr += sizeof(struct nvinfo_attribute_header_t);
                        if (hdr->attribute == NVINFO_EIATTR_PARAM_CBANK && hdr->format == NVINFO_EIFMT_SVAL) {
                            const struct nvinfo_param_cbank_t *attr = (const struct nvinfo_param_cbank_t *)ptr;
                            f->param_base_offset = attr->params_offset ? attr->params_offset : 0x140;
                        } else if (hdr->attribute == NVINFO_EIATTR_KPARAM_INFO && hdr->format == NVINFO_EIFMT_SVAL) {
                            const struct nvinfo_kparam_info_t *attr = (const struct nvinfo_kparam_info_t *)ptr;
                            int idx = attr->ordinal; // Usar attr->ordinal para indexar f->params
                            if (idx < 64) {
                                f->params[idx].ordinal = attr->ordinal;
                                f->params[idx].offset = (f->param_base_offset ? f->param_base_offset : 0x140) + attr->offset;
                                uint16_t sz = (attr->packed >> 18) & 0x3fff;
                                if (sz == 0) sz = 8;
                                f->params[idx].size = sz;
                                if (idx + 1 > f->num_params) f->num_params = idx + 1;
                            }
                        } else if (hdr->attribute == NVINFO_EIATTR_NUM_BARRIERS) {
                            f->num_barriers = (hdr->format == NVINFO_EIFMT_BVAL) ? hdr->bval.value :
                                              (hdr->format == NVINFO_EIFMT_HVAL) ? hdr->hval.value : 0;
                        }
                        if (hdr->format == NVINFO_EIFMT_SVAL) ptr += hdr->sval.size;
                    }
                }
            }
        } else {
            free(f);
            return CUDA_ERROR_NOT_FOUND;
        }
    } else {
        // Módulo SASS plano (fallback): asignar BO propio y clonar código
        int ret = nouveau_bo_new(g_current_ctx->dev, NOUVEAU_BO_VRAM, 256, hmod->code_size, NULL, &f->code_bo);
        if (ret == 0 && f->code_bo) {
            nouveau_bo_map(f->code_bo, NOUVEAU_BO_WR, g_current_ctx->client);
            memcpy(f->code_bo->map, hmod->code_bo->map, hmod->code_size);
        }
    }
    if (f->local_size > 0) {
        size_t frame_aligned = (f->local_size + 0xf) & ~0xf;
        uint32_t crs = 0x800; // Warp call-return stack requirement (Mesa / NVIDIA hardware)
        size_t warp_bytes = 32 * frame_aligned + crs;
        size_t mp_bytes = (64 * warp_bytes + 0x7fff) & ~0x7fff; // Multiple of 32 KiB (0x8000)
        size_t tls_size = (2 * mp_bytes + 0x1ffff) & ~0x1ffff;  // Aligned to 128 KiB (1 << 17)
        f->tls_mp_size = (uint32_t)mp_bytes;
        int ret = nouveau_bo_new(g_current_ctx->dev, NOUVEAU_BO_VRAM, 1 << 17, tls_size, NULL, &f->tls_bo);
        if (ret == 0 && f->tls_bo) {
            nouveau_bo_map(f->tls_bo, NOUVEAU_BO_WR, g_current_ctx->client);
            memset(f->tls_bo->map, 0, tls_size);
        }
    }

    f->next = hmod->funcs;
    hmod->funcs = f;

    printf("[DEBUG CUBIN] func='%s' regs=%u smem=%u local=%u barriers=%u params=%d base_off=0x%x\n",
           name ? name : "(null)", f->num_regs, f->shared_size, f->local_size, f->num_barriers, f->num_params, f->param_base_offset);
    for (int p = 0; p < f->num_params; p++) {
        printf("    param[%d]: ord=%u off=0x%x sz=%u\n", p, f->params[p].ordinal, f->params[p].offset, f->params[p].size);
    }

    *hfunc = f;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    if (!dptr || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
    if (!hmod->is_elf || !hmod->elf_data) return CUDA_ERROR_NOT_FOUND;

    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)hmod->elf_data;
    Elf64_Shdr *shdrs = (Elf64_Shdr *)((uint8_t *)hmod->elf_data + ehdr->e_shoff);
    const char *shstrtab = (const char *)hmod->elf_data + shdrs[ehdr->e_shstrndx].sh_offset;

    for (int i = 0; i < ehdr->e_shnum; i++) {
        if (shdrs[i].sh_type == SHT_SYMTAB) {
            Elf64_Shdr *symtab = &shdrs[i];
            const char *strtab = (const char *)hmod->elf_data + shdrs[symtab->sh_link].sh_offset;
            Elf64_Sym *syms = (Elf64_Sym *)((uint8_t *)hmod->elf_data + symtab->sh_offset);
            int num_syms = symtab->sh_size / sizeof(Elf64_Sym);

            for (int s = 0; s < num_syms; s++) {
                const char *sname = strtab + syms[s].st_name;
                if (strcmp(sname, name) == 0) {
                    int sec_idx = syms[s].st_shndx;
                    if (sec_idx < 0 || sec_idx >= ehdr->e_shnum) return CUDA_ERROR_NOT_FOUND;
                    const char *sec_name = shstrtab + shdrs[sec_idx].sh_name;
                    int bank = -1;
                    if (sscanf(sec_name, ".nv.constant%d", &bank) == 1 && bank >= 1 && bank < 8) {
                        if (hmod->cb_bo[bank]) {
                            *dptr = (CUdeviceptr)(hmod->cb_bo[bank]->offset + syms[s].st_value);
                            if (bytes) *bytes = syms[s].st_size;
                            printf("[DEBUG CONSTANT] Resolved global symbol '%s' -> CB%d (dptr=0x%lx, size=%zu)\n",
                                   name, bank, (unsigned long)*dptr, (size_t)(bytes ? *bytes : 0));
                            return CUDA_SUCCESS;
                        }
                    }
                }
            }
        }
    }
    return CUDA_ERROR_NOT_FOUND;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    return cuModuleGetGlobal(dptr, bytes, hmod, name);
}

/* === LANZAMIENTO DE KERNEL EN SILICIO (cuLaunchKernel) === */

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra)
{
    (void)hStream;
    (void)extra;
    if (!g_current_ctx || !f) return CUDA_ERROR_INVALID_CONTEXT;
    struct CUctx_st *ctx = g_current_ctx;

    if (blockDimX == 0 || blockDimY == 0 || blockDimZ == 0 ||
        gridDimX == 0 || gridDimY == 0 || gridDimZ == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    uint64_t total_threads = (uint64_t)blockDimX * blockDimY * blockDimZ;
    if (total_threads > 1024 || blockDimX > 1024 || blockDimY > 1024 || blockDimZ > 64) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (gridDimY > 65535 || gridDimZ > 65535) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ((uint64_t)sharedMemBytes + f->shared_size > 49152) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    // 1. Solicita Turbo Boost P0 (926 MHz) antes de tocar el hardware
    gpu_governor_request_turbo(3000);

    pthread_mutex_lock(&ctx->mutex);

    // 2. Poblar Constant Buffer 0 (CB0) en VRAM con geometría y argumentos ABI NVIDIA
    nouveau_bo_map(ctx->cb0_bo, NOUVEAU_BO_WR, ctx->client);
    memset(ctx->cb0_bo->map, 0, 4096);
    uint32_t *cb0_ptr = (uint32_t *)ctx->cb0_bo->map;

    // Offsets auxiliares / retrocompatibilidad
    cb0_ptr[0x08 / 4] = blockDimX;
    cb0_ptr[0x0c / 4] = blockDimY;
    cb0_ptr[0x10 / 4] = blockDimZ;
    cb0_ptr[0x14 / 4] = gridDimX;
    cb0_ptr[0x18 / 4] = gridDimY;
    cb0_ptr[0x1c / 4] = gridDimZ;

    // Offsets estándar de la ABI sm_30 de NVIDIA (leídos por ptxas / SASS)
    cb0_ptr[0x20 / 4] = gridDimX;
    cb0_ptr[0x24 / 4] = gridDimY;
    cb0_ptr[0x28 / 4] = blockDimX; // ntid.x
    cb0_ptr[0x2c / 4] = blockDimY; // ntid.y
    cb0_ptr[0x30 / 4] = blockDimZ; // ntid.z
    cb0_ptr[0x34 / 4] = gridDimZ;
    // NVIDIA ABI Kepler SM30: Stack pointer inicial por hilo en c0[0x44].
    // El prólogo del shader ejecuta: mov b32 $r1, c0[0x44]; add b32 $r1, $r1, -frame_size;
    // Por tanto, c0[0x44] debe inicializarse en frame_aligned para que el frame inicie en offset local 0.
    if (f->local_size > 0) {
        size_t frame_aligned = (f->local_size + 0xf) & ~0xf;
        cb0_ptr[0x44 / 4] = (uint32_t)frame_aligned;
    }

    // [PASO 1] Empaquetar argumentos en CB0 de forma exacta según metadatos .nv.info
    if (kernelParams && f->num_params > 0) {
        for (int i = 0; i < f->num_params; i++) {
            uint16_t off = f->params[i].offset;
            uint16_t sz  = f->params[i].size;
            if (off >= 0x140 && off + sz <= 4096 && kernelParams[i] != NULL) {
                memcpy((uint8_t *)cb0_ptr + off, kernelParams[i], sz);
            }
        }
    } else if (kernelParams) {
        // Fallback tolerante para microkernels sin metadatos ELF
        int i = 0;
        while (kernelParams[i] != NULL && i < 16) {
            uint64_t val = *(uint64_t *)kernelParams[i];
            cb0_ptr[(0x140 + i * 8) / 4]     = (uint32_t)(val & 0xffffffff);
            cb0_ptr[(0x140 + i * 8) / 4 + 1] = (uint32_t)(val >> 32);
            i++;
        }
        if (i == 1) {
            uint64_t arg0 = *(uint64_t *)kernelParams[0];
            cb0_ptr[0x148 / 4]       = (uint32_t)(arg0 & 0xffffffff);
            cb0_ptr[(0x148 / 4) + 1] = (uint32_t)(arg0 >> 32);
        }
    }

    // 3. Configurar Descriptor QMD v00_06 (256 bytes) en VRAM
    nouveau_bo_map(ctx->desc_bo, NOUVEAU_BO_WR, ctx->client);
    uint32_t *qmd = (uint32_t *)ctx->desc_bo->map;
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

    // [PASO 2] Dirección del código del kernel en VRAM (aislado por función)
    struct nouveau_bo *target_code_bo = f->code_bo ? f->code_bo : (f->mod ? f->mod->code_bo : NULL);
    NVA0C0_QMDV00_06_VAL_SET(qmd, PROGRAM_OFFSET, (uint32_t)f->code_offset);

    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_WIDTH, gridDimX);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_HEIGHT, gridDimY);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_RASTER_DEPTH, gridDimZ);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION0, blockDimX);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION1, blockDimY);
    NVA0C0_QMDV00_06_VAL_SET(qmd, CTA_THREAD_DIMENSION2, blockDimZ);

    // [PASO 3] Configuración dinámica de memoria compartida y L1
    uint32_t total_smem = f->shared_size + sharedMemBytes;
    uint32_t smem_aligned = (total_smem + 0xff) & ~0xff;

    if (total_smem > 32768) {
        NVA0C0_QMDV00_06_DEF_SET(qmd, L1_CONFIGURATION, DIRECTLY_ADDRESSABLE_MEMORY_SIZE_48KB);
    } else if (total_smem > 16384) {
        NVA0C0_QMDV00_06_DEF_SET(qmd, L1_CONFIGURATION, DIRECTLY_ADDRESSABLE_MEMORY_SIZE_32KB);
    } else {
        NVA0C0_QMDV00_06_DEF_SET(qmd, L1_CONFIGURATION, DIRECTLY_ADDRESSABLE_MEMORY_SIZE_16KB);
    }

    uint32_t regs = f->num_regs ? f->num_regs : 16;
    if (regs > 63) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    }
    NVA0C0_QMDV00_06_VAL_SET(qmd, REGISTER_COUNT, regs);
    NVA0C0_QMDV00_06_VAL_SET(qmd, BARRIER_COUNT, f->num_barriers);
    NVA0C0_QMDV00_06_VAL_SET(qmd, SHARED_MEMORY_SIZE, smem_aligned);

    // [PASO 2] Configuración de Memoria Local (spills a stack) en QMD
    if (f->local_size > 0) {
        size_t frame_aligned = (f->local_size + 0xf) & ~0xf;
        NVA0C0_QMDV00_06_VAL_SET(qmd, SHADER_LOCAL_MEMORY_LOW_SIZE, (uint32_t)frame_aligned);
        NVA0C0_QMDV00_06_VAL_SET(qmd, SHADER_LOCAL_MEMORY_HIGH_SIZE, 0);
        NVA0C0_QMDV00_06_VAL_SET(qmd, SHADER_LOCAL_MEMORY_CRS_SIZE, 0x800);
    }

    // Enlazar CB0 en VRAM
    NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_ADDR_LOWER, 0, (uint32_t)(ctx->cb0_bo->offset & 0xffffffff));
    NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_ADDR_UPPER, 0, (uint32_t)(ctx->cb0_bo->offset >> 32));
    NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_SIZE, 0, 4096);
    NVA0C0_QMDV00_06_DEF_SET(qmd, CONSTANT_BUFFER_VALID, 0, TRUE);

    // [PASO 3] Enlazar Constant Buffers de usuario CB1..CB7 si están presentes en el módulo
    struct CUmod_st *mod = f->mod;
    if (mod) {
        for (int b = 1; b < 8; b++) {
            if (mod->cb_bo[b]) {
                uint32_t aligned_sz = (uint32_t)((mod->cb_size[b] + 0xff) & ~0xff);
                if (aligned_sz < 256) aligned_sz = 256;
                NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_ADDR_LOWER, b, (uint32_t)(mod->cb_bo[b]->offset & 0xffffffff));
                NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_ADDR_UPPER, b, (uint32_t)(mod->cb_bo[b]->offset >> 32));
                NVA0C0_QMDV00_06_VAL_SET(qmd, CONSTANT_BUFFER_SIZE, b, aligned_sz);
                NVA0C0_QMDV00_06_DEF_SET(qmd, CONSTANT_BUFFER_VALID, b, TRUE);
            }
        }
    }

    // 4. Registrar referencias de buffers activos en el MMU
    int num_refs = 0;
    struct nouveau_pushbuf_refn refs[MAX_ALLOCS + 16];

    refs[num_refs++] = (struct nouveau_pushbuf_refn){ target_code_bo, NOUVEAU_BO_VRAM | NOUVEAU_BO_RD };
    refs[num_refs++] = (struct nouveau_pushbuf_refn){ ctx->cb0_bo,    NOUVEAU_BO_VRAM | NOUVEAU_BO_RD };
    refs[num_refs++] = (struct nouveau_pushbuf_refn){ ctx->desc_bo,   NOUVEAU_BO_VRAM | NOUVEAU_BO_RD };
    if (f->tls_bo) {
        refs[num_refs++] = (struct nouveau_pushbuf_refn){ f->tls_bo, NOUVEAU_BO_VRAM | NOUVEAU_BO_RDWR };
    }
    if (ctx->fence_bo) {
        refs[num_refs++] = (struct nouveau_pushbuf_refn){ ctx->fence_bo, NOUVEAU_BO_GART | NOUVEAU_BO_WR };
    }
    if (mod) {
        for (int b = 1; b < 8; b++) {
            if (mod->cb_bo[b]) {
                refs[num_refs++] = (struct nouveau_pushbuf_refn){ mod->cb_bo[b], NOUVEAU_BO_VRAM | NOUVEAU_BO_RD };
            }
        }
    }

    // Agregar todos los buffers de usuario asignados en el contexto
    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (ctx->allocs[i].in_use && ctx->allocs[i].bo) {
            refs[num_refs++] = (struct nouveau_pushbuf_refn){ ctx->allocs[i].bo, NOUVEAU_BO_VRAM | NOUVEAU_BO_RDWR };
        }
    }

    if (nouveau_pushbuf_space(ctx->push, 64, num_refs, 0) != 0) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_ERROR_UNKNOWN;
    }

    // Incrementar secuencia solo tras verificar que hay espacio en el pushbuffer
    uint32_t seq = ++ctx->launch_seq;
    if (hStream) {
        hStream->last_seq = seq;
    }

    // Semáforo de hardware al terminar el kernel
    if (ctx->fence_bo) {
        NVA0C0_QMDV00_06_DEF_SET(qmd, SEMAPHORE_RELEASE_ENABLE0, TRUE);
        NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_ADDRESS_LOWER, (uint32_t)(ctx->fence_bo->offset & 0xffffffff));
        NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_ADDRESS_UPPER, (uint32_t)(ctx->fence_bo->offset >> 32));
        NVA0C0_QMDV00_06_VAL_SET(qmd, RELEASE0_PAYLOAD, seq);
        NVA0C0_QMDV00_06_DEF_SET(qmd, RELEASE0_STRUCTURE_SIZE, ONE_WORD);
    }

    // Garantizar que todas las escrituras a CB0 y QMD en BAR1 estén volcadas
    __builtin_ia32_sfence();

    nouveau_pushbuf_refn(ctx->push, refs, num_refs);

    // 5. Emitir comandos Kepler al pushbuffer
    *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, 0x0000, 1);
    *ctx->push->cur++ = ctx->compute_obj->oclass;

    *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LOCAL_BASE, 1);
    *ctx->push->cur++ = 0xff << 24;
    *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_SHARED_BASE, 1);
    *ctx->push->cur++ = 0xfe << 24;
    *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_UNK0310, 1);
    *ctx->push->cur++ = 0x300;

    // [PASO 2] Configuración de memoria local en el pipeline gráfico
    if (f->tls_bo) {
        uint32_t mp_temp_size = f->tls_mp_size ? f->tls_mp_size : (uint32_t)(f->tls_bo->size / 2);
        *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, 0x0790, 2); // TEMP_ADDRESS_HIGH/LOW
        *ctx->push->cur++ = (uint32_t)(f->tls_bo->offset >> 32);
        *ctx->push->cur++ = (uint32_t)(f->tls_bo->offset & 0xffffffff);

        *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, 0x02e4, 3); // MP_TEMP_SIZE_HIGH(0)
        *ctx->push->cur++ = 0;
        *ctx->push->cur++ = mp_temp_size & ~0x7fff;
        *ctx->push->cur++ = 0xff;

        *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, 0x02f0, 3); // MP_TEMP_SIZE_HIGH(1)
        *ctx->push->cur++ = 0;
        *ctx->push->cur++ = mp_temp_size & ~0x7fff;
        *ctx->push->cur++ = 0xff;
    }

    *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_CODE_ADDRESS_HIGH, 2);
    *ctx->push->cur++ = (uint32_t)(target_code_bo->offset >> 32);
    *ctx->push->cur++ = (uint32_t)(target_code_bo->offset & 0xffffffff);

    // Invalidar cachés
    *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_INVALIDATE_CACHES, 1);
    *ctx->push->cur++ = 0x00001011;

    *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_FLUSH, 1);
    *ctx->push->cur++ = 0x00000001;

    *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LAUNCH_DESC_ADDR, 1);
    *ctx->push->cur++ = (uint32_t)(ctx->desc_bo->offset >> 8);

    *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_LAUNCH, 1);
    *ctx->push->cur++ = 0x00000003;

    *ctx->push->cur++ = NVC0_FIFO_PKHDR_SQ(1, NVE4_COMPUTE_SERIALIZE, 1);
    *ctx->push->cur++ = 0x00000000;

    // Despacho asíncrono
    int kick_ret = nouveau_pushbuf_kick(ctx->push, ctx->chan);

    pthread_mutex_unlock(&ctx->mutex);

    if (kick_ret != 0) return CUDA_ERROR_UNKNOWN;
    return CUDA_SUCCESS;
}

/* === STREAMS CUDA === */

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    return cuStreamCreateWithPriority(phStream, Flags, 0);
}

CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int Flags, int priority) {
    if (!phStream || !g_current_ctx) return CUDA_ERROR_INVALID_VALUE;
    struct CUstream_st *s = (struct CUstream_st *)calloc(1, sizeof(struct CUstream_st));
    if (!s) return CUDA_ERROR_OUT_OF_MEMORY;
    s->ctx = g_current_ctx;
    s->flags = Flags;
    s->priority = priority;
    s->last_seq = 0;
    *phStream = s;
    return CUDA_SUCCESS;
}

CUresult cuStreamDestroy(CUstream hStream) {
    if (!hStream) return CUDA_ERROR_INVALID_HANDLE;
    cuStreamSynchronize(hStream);
    free(hStream);
    return CUDA_SUCCESS;
}

CUresult cuStreamQuery(CUstream hStream) {
    struct CUctx_st *ctx = hStream ? hStream->ctx : g_current_ctx;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    uint32_t target_seq = hStream ? hStream->last_seq : ctx->launch_seq;
    if (target_seq == 0) return CUDA_SUCCESS;
    if (!ctx->fence_bo || !ctx->fence_bo->map) return CUDA_SUCCESS;
    volatile uint32_t *fence_ptr = (volatile uint32_t *)ctx->fence_bo->map;
    if ((int32_t)(*fence_ptr - target_seq) >= 0) return CUDA_SUCCESS;
    return CUDA_ERROR_NOT_READY;
}

CUresult cuStreamSynchronize(CUstream hStream) {
    struct CUctx_st *ctx = hStream ? hStream->ctx : g_current_ctx;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    pthread_mutex_lock(&ctx->mutex);

    uint32_t target_seq = hStream ? hStream->last_seq : ctx->launch_seq;
    if (target_seq == 0) {
        pthread_mutex_unlock(&ctx->mutex);
        return CUDA_SUCCESS;
    }

    if (ctx->push && ctx->chan) {
        nouveau_pushbuf_kick(ctx->push, ctx->chan);
    }

    int wait_err = 0;
    if (ctx->fence_bo && ctx->fence_bo->map) {
        volatile uint32_t *fence_ptr = (volatile uint32_t *)ctx->fence_bo->map;
        struct timespec start_ts, now_ts;
        clock_gettime(CLOCK_MONOTONIC, &start_ts);
        uint64_t deadline_ns = (uint64_t)start_ts.tv_sec * 1000000000ULL + start_ts.tv_nsec + 5000000000ULL;
        while ((int32_t)(*fence_ptr - target_seq) < 0) {
            clock_gettime(CLOCK_MONOTONIC, &now_ts);
            uint64_t now_ns = (uint64_t)now_ts.tv_sec * 1000000000ULL + now_ts.tv_nsec;
            if (now_ns >= deadline_ns) {
                wait_err = nouveau_bo_wait(ctx->fence_bo, NOUVEAU_BO_RDWR, ctx->client);
                break;
            }
            usleep(50);
        }
    }
    pthread_mutex_unlock(&ctx->mutex);

    if (wait_err != 0 || (ctx->fence_bo && (int32_t)(*(volatile uint32_t *)ctx->fence_bo->map - target_seq) < 0)) {
        return CUDA_ERROR_LAUNCH_TIMEOUT;
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    (void)Flags;
    if (!hEvent) return CUDA_ERROR_INVALID_HANDLE;
    if (!hEvent->recorded) return CUDA_SUCCESS;
    struct CUctx_st *ctx = hStream ? hStream->ctx : (hEvent->ctx ? hEvent->ctx : g_current_ctx);
    if (!ctx || !ctx->fence_bo || !ctx->fence_bo->map) return CUDA_SUCCESS;
    volatile uint32_t *fence_ptr = (volatile uint32_t *)ctx->fence_bo->map;
    while ((int32_t)(*fence_ptr - hEvent->seq) < 0) {
        usleep(50);
    }
    return CUDA_SUCCESS;
}

/* === EVENTOS CUDA NATIVOS CON SEMÁFOROS DE HARDWARE === */

CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    if (!phEvent || !g_current_ctx) return CUDA_ERROR_INVALID_VALUE;
    struct CUevent_st *ev = (struct CUevent_st *)calloc(1, sizeof(struct CUevent_st));
    if (!ev) return CUDA_ERROR_OUT_OF_MEMORY;
    ev->ctx = g_current_ctx;
    ev->flags = Flags;
    ev->recorded = 0;
    ev->seq = 0;
    ev->timestamp_ns = 0;
    *phEvent = ev;
    return CUDA_SUCCESS;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    if (!hEvent) return CUDA_ERROR_INVALID_HANDLE;
    struct CUctx_st *ctx = hStream ? hStream->ctx : (g_current_ctx ? g_current_ctx : hEvent->ctx);
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;

    hEvent->ctx = ctx;
    hEvent->seq = hStream ? hStream->last_seq : ctx->launch_seq;
    hEvent->timestamp_ns = get_time_ns();
    hEvent->recorded = 1;

    // Disparar pushbuffer si está pendiente
    if (ctx->push && ctx->chan) {
        nouveau_pushbuf_kick(ctx->push, ctx->chan);
    }
    return CUDA_SUCCESS;
}

CUresult cuEventQuery(CUevent hEvent) {
    if (!hEvent) return CUDA_ERROR_INVALID_HANDLE;
    if (!hEvent->recorded) return CUDA_ERROR_NOT_READY;
    struct CUctx_st *ctx = hEvent->ctx ? hEvent->ctx : g_current_ctx;
    if (!ctx || !ctx->fence_bo || !ctx->fence_bo->map) return CUDA_SUCCESS;
    volatile uint32_t *fence_ptr = (volatile uint32_t *)ctx->fence_bo->map;
    if ((int32_t)(*fence_ptr - hEvent->seq) >= 0) return CUDA_SUCCESS;
    return CUDA_ERROR_NOT_READY;
}

CUresult cuEventSynchronize(CUevent hEvent) {
    if (!hEvent) return CUDA_ERROR_INVALID_HANDLE;
    if (!hEvent->recorded) return CUDA_ERROR_NOT_READY;
    struct CUctx_st *ctx = hEvent->ctx ? hEvent->ctx : g_current_ctx;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;

    if (ctx->push && ctx->chan) {
        nouveau_pushbuf_kick(ctx->push, ctx->chan);
    }

    if (ctx->fence_bo && ctx->fence_bo->map) {
        volatile uint32_t *fence_ptr = (volatile uint32_t *)ctx->fence_bo->map;
        struct timespec start_ts, now_ts;
        clock_gettime(CLOCK_MONOTONIC, &start_ts);
        uint64_t deadline_ns = (uint64_t)start_ts.tv_sec * 1000000000ULL + start_ts.tv_nsec + 5000000000ULL;
        while ((int32_t)(*fence_ptr - hEvent->seq) < 0) {
            clock_gettime(CLOCK_MONOTONIC, &now_ts);
            uint64_t now_ns = (uint64_t)now_ts.tv_sec * 1000000000ULL + now_ts.tv_nsec;
            if (now_ns >= deadline_ns) {
                nouveau_bo_wait(ctx->fence_bo, NOUVEAU_BO_RDWR, ctx->client);
                break;
            }
            usleep(50);
        }
        if (ctx->fence_bo && (int32_t)(*(volatile uint32_t *)ctx->fence_bo->map - hEvent->seq) < 0) {
            return CUDA_ERROR_LAUNCH_TIMEOUT;
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    if (!pMilliseconds || !hStart || !hEnd) return CUDA_ERROR_INVALID_VALUE;
    if (!hStart->recorded || !hEnd->recorded) return CUDA_ERROR_NOT_READY;

    // Sincronizar el evento final si aún no ha completado en el hardware
    CUresult r = cuEventSynchronize(hEnd);
    if (r != CUDA_SUCCESS) return r;

    int64_t diff_ns = (int64_t)(hEnd->timestamp_ns - hStart->timestamp_ns);
    if (diff_ns < 0) diff_ns = 0;
    *pMilliseconds = (float)((double)diff_ns / 1000000.0);
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy(CUevent hEvent) {
    if (!hEvent) return CUDA_ERROR_INVALID_HANDLE;
    free(hEvent);
    return CUDA_SUCCESS;
}
