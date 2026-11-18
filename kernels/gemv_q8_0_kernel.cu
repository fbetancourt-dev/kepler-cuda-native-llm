#include <cuda.h>
#include <stdint.h>

// ====================================================================
// ESTRUCTURA DE BLOQUE Q8_0 (32 pesos int8 + escala float32)
// 36 bytes por bloque de 32 elementos (vs 128 bytes en FP32 puro)
// ====================================================================

typedef struct __align__(4) {
    float d;           // Factor de escala delta del bloque
    int8_t qs[32];     // 32 pesos cuantizados uniformemente en int8
} block_q8_0;

// ====================================================================
// MICRO-KERNEL GEMV Q8_0 VECTORIZADO (KEPLER SM_30 NATIVO)
// 1 Warp (32 hilos) por cada fila de la matriz
// Lectura coalescida de pesos int8 y reducción SHFL.DOWN
// ====================================================================

extern "C" __global__ void gemv_q8_0_kernel(
    float * __restrict__ y,
    const block_q8_0 * __restrict__ A,  // Matriz de bloques Q8_0: M filas x num_blocks
    const float * __restrict__ x,       // Vector de activación en FP32: K elementos
    int M,                              // Número de filas
    int num_blocks)                     // Número de bloques por fila (num_blocks = K / 32)
{
    int lane_id = threadIdx.x;          // 0 .. 31 dentro del warp
    int warp_id = threadIdx.y;          // Índice de warp dentro del bloque
    int row = blockIdx.x * blockDim.y + warp_id;

    if (row >= M) return;

    const block_q8_0 *row_blocks = A + (size_t)row * num_blocks;
    float row_acc = 0.0f;

    // Iterar sobre todos los bloques de la fila
    for (int b = 0; b < num_blocks; b++) {
        // Cada hilo 'lane_id' lee exactamente 1 peso int8 y 1 activación FP32
        int8_t q_val = row_blocks[b].qs[lane_id];
        float x_val = x[b * 32 + lane_id];

        // Producto escalar parcial
        float prod = (float)q_val * x_val;

        // Reducción intra-warp con SHFL nativo de Kepler sm_30
        for (int offset = 16; offset > 0; offset /= 2) {
            prod += __shfl_down_sync(0xffffffff, prod, offset);
        }

        // El hilo 0 del warp lee la escala del bloque y multiplica la suma
        if (lane_id == 0) {
            row_acc += prod * row_blocks[b].d;
        }
    }

    // El hilo 0 escribe el resultado final de la fila en VRAM
    if (lane_id == 0) {
        y[row] = row_acc;
    }
}
