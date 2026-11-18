extern "C" __global__ void rms_norm_kernel(
    float * __restrict__ y,
    const float * __restrict__ x,
    const float * __restrict__ w,
    int hidden_dim,
    float eps)
{
    // Memoria compartida dinámica para la reducción de sum(x^2)
    extern __shared__ float s_data[];

    int row = blockIdx.x;
    const float *x_row = x + (size_t)row * hidden_dim;
    float *y_row = y + (size_t)row * hidden_dim;

    // 1. Acumulación paralela de cuadrados por hilo
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = x_row[i];
        sum_sq += val * val;
    }
    s_data[threadIdx.x] = sum_sq;
    __syncthreads();

    // 2. Reducción en árbol en shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // 3. Hilo 0 calcula el factor de escala inverso (rsqrt)
    if (threadIdx.x == 0) {
        float mean_sq = s_data[0] / (float)hidden_dim;
        s_data[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float rrms = s_data[0];

    // 4. Normalizar y escalar por los pesos w[i]
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        y_row[i] = x_row[i] * rrms * w[i];
    }
}
