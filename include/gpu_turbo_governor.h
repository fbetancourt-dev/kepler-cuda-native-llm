#ifndef __GPU_TURBO_GOVERNOR_H__
#define __GPU_TURBO_GOVERNOR_H__

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

int gpu_governor_client_init(void);
int gpu_governor_request_turbo(uint32_t ttl_ms);
int gpu_governor_renew_turbo(uint32_t ttl_ms);
int gpu_governor_release_turbo(void);
void gpu_governor_client_close(void);

#ifdef __cplusplus
}
#endif

#endif // __GPU_TURBO_GOVERNOR_H__
