extern "C" __global__ void rope_kcache_kernel(
    float * __restrict__ Q,               // [num_heads, head_dim]
    const float * __restrict__ K_in,     // [num_kv_heads, head_dim]
    const float * __restrict__ V_in,     // [num_kv_heads, head_dim]
    const float * __restrict__ cos_pos,  // cosenos precalculados para 'pos' [head_dim / 2]
    const float * __restrict__ sin_pos,  // senos precalculados para 'pos' [head_dim / 2]
    float * __restrict__ K_cache,        // [num_kv_heads, max_seq_len, head_dim]
    float * __restrict__ V_cache,        // [num_kv_heads, max_seq_len, head_dim]
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int mode)                            // 0: rotate_half (HuggingFace/Llama), 1: interleaved (Meta)
{
    int h = blockIdx.x;
    int half_dim = head_dim / 2;
    int tid = threadIdx.x;

    if (tid >= half_dim) return;

    float c = cos_pos[tid];
    float s = sin_pos[tid];

    int idx0, idx1;
    if (mode == 0) {
        // HuggingFace / Transformers rotate_half: pares (i, i + half_dim)
        idx0 = tid;
        idx1 = tid + half_dim;
    } else {
        // Meta LLaMA interleaved: pares (2*i, 2*i + 1)
        idx0 = tid * 2;
        idx1 = idx0 + 1;
    }

    // 1. Aplicar RoPE a Query (Q) para todas las cabezas Q
    if (h < num_heads) {
        float *q_head = Q + (size_t)h * head_dim;
        float q0 = q_head[idx0];
        float q1 = q_head[idx1];

        if (mode == 0) {
            // rotate_half: q_rot = [q0*c - q1*s, q1*c + q0*s]
            q_head[idx0] = q0 * c - q1 * s;
            q_head[idx1] = q1 * c + q0 * s;
        } else {
            // interleaved: q_rot = [q0*c - q1*s, q0*s + q1*c]
            q_head[idx0] = q0 * c - q1 * s;
            q_head[idx1] = q0 * s + q1 * c;
        }
    }

    // 2. Aplicar RoPE a Key (K) y escribir simultáneamente en K_cache y V_cache
    if (h < num_kv_heads) {
        const float *k_head_in = K_in + (size_t)h * head_dim;
        const float *v_head_in = V_in + (size_t)h * head_dim;

        float k0 = k_head_in[idx0];
        float k1 = k_head_in[idx1];

        float k_rot0, k_rot1;
        if (mode == 0) {
            k_rot0 = k0 * c - k1 * s;
            k_rot1 = k1 * c + k0 * s;
        } else {
            k_rot0 = k0 * c - k1 * s;
            k_rot1 = k0 * s + k1 * c;
        }

        size_t cache_offset = ((size_t)h * max_seq_len + pos) * head_dim;
        K_cache[cache_offset + idx0] = k_rot0;
        K_cache[cache_offset + idx1] = k_rot1;

        V_cache[cache_offset + idx0] = v_head_in[idx0];
        V_cache[cache_offset + idx1] = v_head_in[idx1];
    }
}
