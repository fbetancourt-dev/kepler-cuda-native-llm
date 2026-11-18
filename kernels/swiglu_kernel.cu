extern "C" __global__ void swiglu_kernel(
    float * __restrict__ out,              // [dim]
    const float * __restrict__ gate,       // [dim]
    const float * __restrict__ up,         // [dim]
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    float g = gate[idx];
    float u = up[idx];

    // SiLU(g) = g / (1.0f + expf(-g))
    float silu_g = g / (1.0f + __expf(-g));

    out[idx] = silu_g * u;
}

extern "C" __global__ void residual_add_kernel(
    float * __restrict__ x,                // in-place residual accumulator
    const float * __restrict__ residual,   // delta to add
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    x[idx] += residual[idx];
}
