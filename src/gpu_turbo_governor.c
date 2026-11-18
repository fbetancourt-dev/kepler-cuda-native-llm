#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <glob.h>
#include <pthread.h>
#include <sys/time.h>
#include "gpu_turbo_governor.h"

static char g_temp_path[256] = {0};
static char g_volt_path[256] = {0};
static char g_pstate_path[256] = {0};

static gpu_power_state_t g_current_state = GPU_STATE_IDLE;
static pthread_mutex_t g_gov_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_watchdog_thread;
static volatile bool g_running = false;
static int g_turbo_ref_count = 0;

static uint64_t g_turbo_start_time = 0;
static uint64_t g_turbo_activations = 0;
static uint64_t g_total_turbo_ms = 0;

static uint64_t get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + (uint64_t)tv.tv_usec / 1000;
}

static int read_int_file(const char *path, long *out_val) {
    if (!path || !path[0]) return -1;
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (fscanf(f, "%ld", out_val) != 1) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

static void *watchdog_loop(void *arg) {
    (void)arg;
    while (g_running) {
        pthread_mutex_lock(&g_gov_mutex);

        long temp_mdeg = 0;
        long volt_mv = 0;
        read_int_file(g_temp_path, &temp_mdeg);
        read_int_file(g_volt_path, &volt_mv);

        double temp_c = (double)temp_mdeg / 1000.0;

        if (g_current_state == GPU_STATE_TURBO) {
            // Protección térmica de lazo cerrado (Thermal Guard)
            if (temp_c >= 82.0) {
                // Alerta térmica: Throttle preventivo a estado medio
                g_current_state = GPU_STATE_THROTTLED;
                printf("[!] THERMAL GUARD: Temperatura alcanzo %.1f C. Throttling preventivo activado.\n", temp_c);
            }
        } else if (g_current_state == GPU_STATE_THROTTLED) {
            // Histeresis: solo recupera si baja de 75°C
            if (temp_c < 75.0 && g_turbo_ref_count > 0) {
                g_current_state = GPU_STATE_TURBO;
                printf("[+] THERMAL GUARD: Temperatura recuperada a %.1f C. Reanudando Turbo.\n", temp_c);
            }
        }

        pthread_mutex_unlock(&g_gov_mutex);
        usleep(100000); // Muestreo cada 100 ms
    }
    return NULL;
}

int gpu_governor_init(void) {
    pthread_mutex_lock(&g_gov_mutex);
    if (g_running) {
        pthread_mutex_unlock(&g_gov_mutex);
        return 0;
    }

    // Buscar ruta de hwmon dinámica
    glob_t glob_res;
    if (glob("/sys/class/drm/card0/device/hwmon/hwmon*/temp1_input", 0, NULL, &glob_res) == 0 && glob_res.gl_pathc > 0) {
        strncpy(g_temp_path, glob_res.gl_pathv[0], sizeof(g_temp_path) - 1);
        globfree(&glob_res);
    }

    if (glob("/sys/class/drm/card0/device/hwmon/hwmon*/in0_input", 0, NULL, &glob_res) == 0 && glob_res.gl_pathc > 0) {
        strncpy(g_volt_path, glob_res.gl_pathv[0], sizeof(g_volt_path) - 1);
        globfree(&glob_res);
    }

    // Ruta de pstate
    strncpy(g_pstate_path, "/sys/kernel/debug/dri/0/pstate", sizeof(g_pstate_path) - 1);

    g_current_state = GPU_STATE_IDLE;
    g_turbo_ref_count = 0;
    g_running = true;

    pthread_create(&g_watchdog_thread, NULL, watchdog_loop, NULL);
    pthread_mutex_unlock(&g_gov_mutex);

    return 0;
}

int gpu_governor_request_turbo(void) {
    pthread_mutex_lock(&g_gov_mutex);
    g_turbo_ref_count++;

    if (g_turbo_ref_count == 1) {
        long temp_mdeg = 0;
        read_int_file(g_temp_path, &temp_mdeg);
        double temp_c = (double)temp_mdeg / 1000.0;

        if (temp_c < 82.0) {
            g_current_state = GPU_STATE_TURBO;
            g_turbo_start_time = get_time_ms();
            g_turbo_activations++;

            // Intento de elevar el P-State si debugfs es accesible
            FILE *f = fopen(g_pstate_path, "w");
            if (f) {
                fprintf(f, "0f\n"); // 0f = Nivel máximo Boost
                fclose(f);
            }
        } else {
            g_current_state = GPU_STATE_THROTTLED;
        }
    }

    pthread_mutex_unlock(&g_gov_mutex);
    return (g_current_state == GPU_STATE_TURBO) ? 0 : -1;
}

int gpu_governor_release_turbo(void) {
    pthread_mutex_lock(&g_gov_mutex);
    if (g_turbo_ref_count > 0) {
        g_turbo_ref_count--;
    }

    if (g_turbo_ref_count == 0) {
        if (g_turbo_start_time > 0) {
            uint64_t elapsed = get_time_ms() - g_turbo_start_time;
            g_total_turbo_ms += elapsed;
            g_turbo_start_time = 0;
        }
        g_current_state = GPU_STATE_IDLE;

        // Regreso inmediato al reloj base/reposo (07 o auto)
        FILE *f = fopen(g_pstate_path, "w");
        if (f) {
            fprintf(f, "auto\n");
            fclose(f);
        }
    }

    pthread_mutex_unlock(&g_gov_mutex);
    return 0;
}

int gpu_governor_get_telemetry(gpu_telemetry_t *out) {
    if (!out) return -1;
    pthread_mutex_lock(&g_gov_mutex);

    long temp_mdeg = 0;
    long volt_mv = 0;
    read_int_file(g_temp_path, &temp_mdeg);
    read_int_file(g_volt_path, &volt_mv);

    out->temp_c = (double)temp_mdeg / 1000.0;
    out->volt_mv = (uint32_t)volt_mv;
    out->state = g_current_state;
    out->max_safe_volt = 1050; // Techo seguro de 1.05V (Límite fábrica: 1.10V)
    out->max_safe_temp = 82.0; // Umbral de protección térmica (Fábrica: 95.0°C)
    out->turbo_activations = g_turbo_activations;
    out->total_turbo_ms = g_total_turbo_ms;

    pthread_mutex_unlock(&g_gov_mutex);
    return 0;
}

void gpu_governor_shutdown(void) {
    if (!g_running) return;
    g_running = false;
    pthread_join(g_watchdog_thread, NULL);
}
