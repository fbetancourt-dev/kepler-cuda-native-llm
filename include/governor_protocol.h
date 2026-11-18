#ifndef __GOVERNOR_PROTOCOL_H__
#define __GOVERNOR_PROTOCOL_H__

#include <stdint.h>
#include <stdbool.h>

#define GOVERNOR_SOCKET_PATH "/tmp/gpu_kepler_governor.sock"
#define GOV_PROTOCOL_VERSION 3

typedef enum {
    GOV_CMD_REQUEST_LEASE = 1,  // Solicitar aceleración con lease temporal
    GOV_CMD_RELEASE_LEASE = 2,  // Liberar aceleración al terminar cómputo
    GOV_CMD_RENEW_LEASE   = 3,  // Heartbeat para kernels largos (>1s)
    GOV_CMD_GET_TELEMETRY = 4,  // Consultar estado, temperaturas y voltajes
    GOV_CMD_PING          = 5   // Verificación de liveness
} gov_cmd_t;

typedef enum {
    STATE_COLD_IDLE_P8  = 0,  // 135 MHz / 324 MHz GDDR5 (0 contextos CUDA, reposo frío)
    STATE_WARM_IDLE_P5  = 1,  // 405 MHz / 2500 MHz GDDR5 (MPLL anclado, contextos vivos)
    STATE_PRE_BOOST_RAMP= 2,  // 650 MHz, rampa escalonada VRM pre-PLL (+150-5000 µs)
    STATE_ACTIVE_TURBO_P0=3,  // 926 MHz Boost, 1050 mV (Cómputo pleno en 0xa0c0)
    STATE_DRAIN_COOLDOWN= 4,  // Dwell Time (300 ms anti-fatiga BGA) + Drenaje PFIFO
    STATE_SAFE_THROTTLE = 5,  // 600 MHz / 2500 MHz GDDR5 (Clamp preventivo >=82C / TTL)
    STATE_FAULT_RECOVERY= 6   // Estado de fallo/recuperación ante error de driver/hardware
} gov_gpu_state_t;

// Códigos de status devueltos al cliente
#define GOV_STATUS_SUCCESS         0   // Lease concedido en P0
#define GOV_STATUS_CLAMPED         1   // Concedido pero clampeado a P5/THROTTLE por calor
#define GOV_STATUS_ERR_NO_LEASE    2   // Aviso idempotente: no poseía lease
#define GOV_STATUS_ERR_SENSOR     -1   // Falla de sensor térmico (Fail-Closed)
#define GOV_STATUS_ERR_INTERNAL   -2   // Error interno de driver o socket

typedef struct {
    uint32_t proto_version;     // GOV_PROTOCOL_VERSION
    uint32_t cmd;               // gov_cmd_t
    uint32_t request_id;        // ID de correlación
    uint32_t lease_ms;          // Duración del lease solicitada (ej. 3000 ms)
} gov_request_t;

typedef struct {
    uint32_t proto_version;     // GOV_PROTOCOL_VERSION
    int32_t  status;            // GOV_STATUS_*
    uint32_t active_state;      // gov_gpu_state_t
    uint32_t daemon_epoch;      // Generación del daemon (detecta reinicios)
    double   temp_gpu_c;        // Temperatura die GPU (C)
    double   temp_cpu_c;        // Temperatura die CPU Haswell (C)
    double   temp_eff_c;        // Temperatura acoplada efectiva (Fórmula Argon)
    uint32_t volt_gpu_mv;       // Voltaje GPU (mV)
    uint32_t fan1_rpm;          // Ventilador izquierdo (Apple SMC)
    uint32_t fan2_rpm;          // Ventilador derecho (Apple SMC)
    uint64_t active_clients;    // Clientes con lease activo
    uint64_t total_turbo_ms;    // Tiempo acumulado en P0
    uint64_t total_bursts;      // Total de ráfagas concedidas
    uint32_t smc_capped;        // Bandera de límite de potencia por SMC/MagSafe
} gov_response_t;

#endif // __GOVERNOR_PROTOCOL_H__
