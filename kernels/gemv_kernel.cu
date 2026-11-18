extern "C" __global__ void gemv_kernel(
    float * __restrict__ y,
    const float4 * __restrict__ A,  // Matriz M x (K/4) de float4
    const float4 * __restrict__ x,  // Vector K/4 de float4
    int M,
    int K4) // K4 = K / 4
{
    // Cada bloque computa un conjunto de filas
    // blockDim.x = 32 (1 warp por fila), blockDim.y = filas por bloque
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int row = blockIdx.x * blockDim.y + warp_id;

    if (row >= M) return;

    const float4 *A_row = A + (size_t)row * K4;

    float sum = 0.0f;
    // Cargas vectorizadas de 128 bits (LD.128 / float4) coalescidas por warp
    for (int col = lane_id; col < K4; col += 32) {
        float4 a_val = A_row[col];
        float4 x_val = x[col];
        sum += a_val.x * x_val.x +
               a_val.y * x_val.y +
               a_val.z * x_val.z +
               a_val.w * x_val.w;
    }

    // Reduccion paralela intra-warp via SHFL nativo Kepler sm_30
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // El primer hilo del warp escribe el resultado de la proyeccion
    if (lane_id == 0) {
        y[row] = sum;
    }
}
