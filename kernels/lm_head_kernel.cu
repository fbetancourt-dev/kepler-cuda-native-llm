extern "C" __global__ void lm_head_kernel(
    float * __restrict__ logits,          // [vocab_size] (ej. 32000 floats)
    const float4 * __restrict__ W_lm,     // [vocab_size, D/4] de float4
    const float4 * __restrict__ x,        // [D/4] de float4
    int vocab_size,
    int K4)                               // D / 4 = 512
{
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int row = blockIdx.x * blockDim.y + warp_id;

    if (row >= vocab_size) return;

    const float4 *W_row = W_lm + (size_t)row * K4;

    float sum = 0.0f;
    for (int col = lane_id; col < K4; col += 32) {
        float4 w_val = W_row[col];
        float4 x_val = x[col];
        sum += w_val.x * x_val.x +
               w_val.y * x_val.y +
               w_val.z * x_val.z +
               w_val.w * x_val.w;
    }

    // Reduccion paralela intra-warp
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        logits[row] = sum;
    }
}

// Reduccion de Argmax rapida en GPU
extern "C" __global__ void argmax_kernel(
    int * __restrict__ out_token_id,      // [1] entero con el token id ganador
    const float * __restrict__ logits,    // [vocab_size]
    int vocab_size)
{
    __shared__ float s_max[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;
    float local_max = -1e38f;
    int local_idx = -1;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    s_max[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Reduccion en shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_token_id = s_idx[0];
    }
}
