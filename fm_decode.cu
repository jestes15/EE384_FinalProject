__global__ void Hilbert(cufftComplex *dev_complex, const int LIMIT_1, const int LIMIT)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x + 1, offset = gridDim.x * blockDim.x;
    while (tid < LIMIT)
    {
        if (tid < LIMIT_1 - 1)
        {
            dev_complex[tid].x *= 2;
            dev_complex[tid].y *= 2;
        }
        if (tid > LIMIT_1 - 1)
        {
            dev_complex[tid].x = 0;
            dev_complex[tid].y = 0;
        }
        tid += offset;
    }
}