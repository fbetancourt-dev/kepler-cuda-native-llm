extern "C" __global__ void attention_decode_kernel(
    float * __restrict__ out,              // [num_heads, head_dim] (ej. 32 x 64 = 2048 floats)
    const float * __restrict__ Q,          // [num_heads, head_dim] (Query actual)
    const float * __restrict__ K_cache,    // [num_kv_heads, max_seq_len, head_dim]
    const float * __restrict__ V_cache,    // [num_kv_heads, max_seq_len, head_dim]
    int num_heads,
    int num_kv_heads,
    int head_dim,                          // Especializado contractualmente a head_dim = 64
    int max_seq_len,
    int context_len,                       // Tokens actuales en el cache (T >= 1)
    float scale)                           // 1.0f / sqrtf(head_dim)
{
    // 1 bloque por cada cabeza de Query (gridDim.x = num_heads = 32)
    // 32 hilos por bloque (1 warp exacto)
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int tid = threadIdx.x; // [0..31]
    int gqa_ratio = num_heads / num_kv_heads;
    int kv_h = h / gqa_ratio;

    const float *q_head = Q + (size_t)h * head_dim;

    // Cada hilo mantiene 2 componentes del vector head_dim (32 hilos * 2 = 64)
    int idx0 = tid * 2;
    int idx1 = idx0 + 1;

    float q0 = q_head[idx0];
    float q1 = q_head[idx1];

    // Acumuladores de Online Softmax en registros locales inicializados a -infinito
    float max_score = -1e38f;
    float sum_exp = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Recorrido de los T tokens en el KV-Cache
    for (int t = 0; t < context_len; t++) {
        size_t cache_off = ((size_t)kv_h * max_seq_len + t) * head_dim;
        const float *k_vec = K_cache + cache_off;
        const float *v_vec = V_cache + cache_off;

        // 1. Dot product parcial (Q * K)
        float dot_part = (q0 * k_vec[idx0] + q1 * k_vec[idx1]);

        // Reduccion completa dentro del warp
        for (int offset = 16; offset > 0; offset /= 2) {
            dot_part += __shfl_down_sync(0xffffffff, dot_part, offset);
        }

        // Difundir el score total S_t a todos los 32 hilos del warp
        float score = __shfl_sync(0xffffffff, dot_part, 0) * scale;

        // 2. Actualizacion de Online Softmax con guarda robusta
        float new_max = (score > max_score) ? score : max_score;
        float alpha = (max_score > -1e37f) ? __expf(max_score - new_max) : 0.0f;
        float beta = __expf(score - new_max);

        sum_exp = sum_exp * alpha + beta;
        acc0 = acc0 * alpha + beta * v_vec[idx0];
        acc1 = acc1 * alpha + beta * v_vec[idx1];

        max_score = new_max;
    }

    // 3. Normalizacion final y volcado a VRAM
    float inv_sum = (sum_exp > 1e-12f) ? (1.0f / sum_exp) : 0.0f;
    float *out_head = out + (size_t)h * head_dim;
    out_head[idx0] = acc0 * inv_sum;
    out_head[idx1] = acc1 * inv_sum;
}
