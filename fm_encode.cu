#include <chrono>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
// #include <matplot/matplot.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <vector>

#include "helper_cuda.h"

float total_time = 0;

#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                                                                               \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>(call);                                                                  \
        if (status != CUFFT_SUCCESS)                                                                                   \
            fprintf(stderr,                                                                                            \
                    "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                           \
                    "with "                                                                                            \
                    "code (%d).\n",                                                                                    \
                    #call, __LINE__, __FILE__, status);                                                                \
    }
#endif // CUFFT_CALL

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>(call);                                                                  \
        if (status != cudaSuccess)                                                                                     \
            fprintf(stderr,                                                                                            \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                         \
                    "with "                                                                                            \
                    "%s (%d).\n",                                                                                      \
                    #call, __LINE__, __FILE__, cudaGetErrorString(status), status);                                    \
    }
#endif // CUDA_RT_CALL

std::vector<std::complex<float>> encode(std::vector<std::complex<float>> input_signal)
{
    cufftHandle plan;
    cudaStream_t stream = NULL;

    std::vector<std::complex<float>> output_signal(input_signal.size());

    cufftComplex *d_data = nullptr;

    checkCudaErrors(cufftCreate(&plan));
    checkCudaErrors(cufftPlan1d(&plan, input_signal.size(), CUFFT_C2C, 1));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cufftSetStream(plan, stream));

    // Create device data arrays
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(std::complex<float>) * input_signal.size()));
    checkCudaErrors(cudaMemcpyAsync(d_data, input_signal.data(), sizeof(std::complex<float>) * input_signal.size(),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    checkCudaErrors(cudaMemcpyAsync(output_signal.data(), d_data, sizeof(std::complex<float>) * output_signal.size(),
                                    cudaMemcpyDeviceToHost, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cufftDestroy(plan));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaDeviceReset());

    return output_signal;
}

#define block_size 32 // 32x32 threads per block -> 1024 threads per block

__global__ void linspace(float *output, int size, float begin_val, float inc_value)
{
    int32_t threadIdx_x = threadIdx.x;
    int32_t threadIdx_y = threadIdx.y;

    int32_t idx = blockDim.x * blockIdx.x + threadIdx_x;
    int32_t idy = blockDim.y * blockIdx.y + threadIdx_y;

    int32_t stride_x = blockDim.x * gridDim.x;
    int32_t stride_y = blockDim.y * gridDim.y;

    for (int j = idy; j < (size >> 1); j += stride_y)
    {
        for (int i = idx; i < (size >> 1); i += stride_x)
        {
            int index = j * block_size + i;
            if (index < size)
                output[index] = index * inc_value + begin_val;
        }
    }
}

__global__ void fill_vector(float *output_signal, float *time, int size, float f1, float f2, float f3, float amplitude1,
                            float amplitude2, float amplitude3)
{
    int32_t threadIdx_x = threadIdx.x;
    int32_t threadIdx_y = threadIdx.y;

    int32_t idx = blockDim.x * blockIdx.x + threadIdx_x;
    int32_t idy = blockDim.y * blockIdx.y + threadIdx_y;

    int32_t stride_x = blockDim.x * gridDim.x;
    int32_t stride_y = blockDim.y * gridDim.y;

    float f1_precomputed = 2.0f * M_PI * f1;
    float f2_precomputed = 2.0f * M_PI * f2;
    float f3_precomputed = 2.0f * M_PI * f3;

    for (int j = idy; j < (size >> 1); j += stride_y)
    {
        for (int i = idx; i < (size >> 1); i += stride_x)
        {
            int index = j * block_size + i;

            if (index < size)
            {
                float temp = time[index];
                output_signal[index] = amplitude1 * __sinf(f1_precomputed * temp) +
                                       amplitude2 * __sinf(f2_precomputed * temp) +
                                       amplitude3 * __sinf(f3_precomputed * temp);
            }
        }
    }
}

__global__ void generate_encoded_signal(float *__restrict__ output, const float *__restrict__ const time,
                                        const float *__restrict__ const cumulative_sum, const float time_constant,
                                        const float cumulative_sum_constant, const unsigned int size,
                                        const float sampling_frequency)
{
    const unsigned int block_x = blockDim.x;
    const unsigned int block_y = blockDim.y;

    const unsigned int local_size = size;

    unsigned int idx = block_x * blockIdx.x + threadIdx.x;
    unsigned int idy = block_y * blockIdx.y + threadIdx.y;

    const unsigned int stride_x = block_x * gridDim.x;
    const unsigned int stride_y = block_y * gridDim.y;

    while (idy < local_size)
    {
        while (idx < local_size)
        {
            const unsigned int index = idy * block_size + idx;
            idx += stride_x;

            if (index < size << 1)
            {
                const float temp1 = time[index];
                const float temp2 = cumulative_sum[index];

                float cos_temp1_temp, sin_temp1_temp;
                float cos_temp2_temp, sin_temp2_temp;

                __sincosf(temp1 * time_constant, &sin_temp1_temp, &cos_temp1_temp);
                __sincosf(temp2 / sampling_frequency * cumulative_sum_constant, &sin_temp2_temp, &cos_temp2_temp);

                output[index] = cos_temp1_temp * cos_temp2_temp - sin_temp1_temp * sin_temp2_temp;
            }
        }
        idy += stride_y;
    }
}

__global__ void hilbert_transform_internals(cuFloatComplex *output, const int LIMIT_1, const int LIMIT)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int offset = gridDim.x * blockDim.x;

    while (tid < LIMIT)
    {
        if (tid < LIMIT_1 - 1)
        {
            output[tid].x *= 2;
            output[tid].y *= 2;
        }
        if (tid > LIMIT_1 - 1)
        {
            output[tid].x = 0;
            output[tid].y = 0;
        }
        tid += offset;
    }
}

__global__ void float_to_complex(float *input, cuFloatComplex *output, int size)
{
    int32_t threadIdx_x = threadIdx.x;
    int32_t threadIdx_y = threadIdx.y;

    int32_t idx = blockDim.x * blockIdx.x + threadIdx_x;
    int32_t idy = blockDim.y * blockIdx.y + threadIdx_y;

    int32_t stride_x = blockDim.x * gridDim.x;
    int32_t stride_y = blockDim.y * gridDim.y;

    for (int j = idy; j < (size >> 1); j += stride_y)
    {
        for (int i = idx; i < (size >> 1); i += stride_x)
        {
            int index = j * block_size + i;
            if (index < size)
            {
                output[index].x = input[index];
                output[index].y = 0.0f;
            }
        }
    }
}

__global__ void yq_computation(cufftComplex *output, cufftComplex *hilbert_transform_data, float *time, float exp_const,
                               int size)
{
    int32_t threadIdx_x = threadIdx.x;
    int32_t threadIdx_y = threadIdx.y;

    int32_t idx = blockDim.x * blockIdx.x + threadIdx_x;
    int32_t idy = blockDim.y * blockIdx.y + threadIdx_y;

    int32_t stride_x = blockDim.x * gridDim.x;
    int32_t stride_y = blockDim.y * gridDim.y;

    int corrected_size = size >> 1;

    for (; idy < corrected_size; idy += stride_y)
    {
        for (; idx < corrected_size; idx += stride_x)
        {
            int index = idy * block_size + idx;

            if (index < size)
            {
                cufftComplex temp = hilbert_transform_data[index];
                float temp_time = time[index];

                output[index].x = temp.x * __cosf(exp_const * temp_time) + temp.y * __sinf(exp_const * temp_time);
                output[index].y = temp.y * __cosf(exp_const * temp_time) - temp.x * __sinf(exp_const * temp_time);
            }
        }
    }
}

__global__ void phase_calculation_and_unwrapping()
{
}

void generate_signal(thrust::device_vector<float> &signal, thrust::device_vector<float> &time, float begin, float end,
                     float sampling_frequency, float f1, float f2, float f3, float amplitude1, float amplitude2,
                     float amplitude3)
{
    int len = (end - begin) / (1.0f / (float)sampling_frequency) + 1;

    if (signal.size() != len)
    {
        signal.resize(len);
    }

    if (time.size() != len)
    {
        time.resize(len);
    }

    dim3 block(block_size, block_size, 1);
    dim3 grid(((signal.size() / 2) + block.x - 1) / block.x, ((signal.size() / 2) + block.y - 1) / block.y);
    linspace<<<grid, block>>>(thrust::raw_pointer_cast(time.data()), time.size(), begin, 1.0f / sampling_frequency);
    fill_vector<<<grid, block>>>(thrust::raw_pointer_cast(signal.data()), thrust::raw_pointer_cast(time.data()),
                                 signal.size(), f1, f2, f3, amplitude1, amplitude2, amplitude3);
}

thrust::device_vector<float> encode(thrust::device_vector<float> signal, thrust::device_vector<float> time,
                                    float sampling_frequency, float carrier_frequency, float frequency_deviation)
{
    thrust::device_vector<float> cumsum_result(signal.size());
    thrust::device_vector<float> encoded_signal(signal.size());
    dim3 block(block_size, block_size, 1);
    dim3 grid(((signal.size() / 2) + block.x - 1) / block.x, ((signal.size() / 2) + block.y - 1) / block.y);

#ifndef NSYS_COMP
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif

    thrust::inclusive_scan(signal.begin(), signal.end(), cumsum_result.begin());
    generate_encoded_signal<<<grid, block>>>(
        thrust::raw_pointer_cast(encoded_signal.data()), thrust::raw_pointer_cast(time.data()),
        thrust::raw_pointer_cast(cumsum_result.data()), 2.0f * M_PI * carrier_frequency,
        2.0f * M_PI * frequency_deviation, encoded_signal.size() >> 1, sampling_frequency);

#ifndef NSYS_COMP
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_time += milliseconds;
#endif
    return encoded_signal;
}

thrust::device_vector<float> decode(thrust::device_vector<float> signal, thrust::device_vector<float> time,
                                    float sampling_frequency, float carrier_frequency, float frequency_deviation)
{
    thrust::device_vector<float> encoded_signal(signal.size());

    cufftComplex *output, *input;
    cufftHandle fwd_plan, inv_plan;
    CUFFT_CALL(cufftPlan1d(&fwd_plan, signal.size(), CUFFT_C2C, 1));
    CUFFT_CALL(cufftPlan1d(&inv_plan, signal.size(), CUFFT_C2C, 1));
    CUDA_RT_CALL(cudaMallocManaged(&input, signal.size() * sizeof(input[0])));
    CUDA_RT_CALL(cudaMallocManaged(&output, signal.size() * sizeof(output[0])));
    dim3 block_2d(block_size, block_size, 1);
    dim3 grid_2d(((signal.size() / 2) + block_2d.x - 1) / block_2d.x,
                 ((signal.size() / 2) + block_2d.y - 1) / block_2d.y);

    float_to_complex<<<grid_2d, block_2d>>>(thrust::raw_pointer_cast(signal.data()), input, signal.size());

    dim3 block(block_size * block_size);
    dim3 grid((signal.size() + block.x - 1) / block.x);

    CUFFT_CALL(cufftExecC2C(fwd_plan, input, output, CUFFT_FORWARD));
    hilbert_transform_internals<<<grid, block>>>(output, signal.size() / 2 + 1, signal.size());
    CUFFT_CALL(cufftExecC2C(inv_plan, output, output, CUFFT_INVERSE));
    CUDA_RT_CALL(cudaDeviceSynchronize());

    return encoded_signal;
}

int main()
{
#ifndef NSYS_COMP
    int frequencies = 10;
    float sampling_frequencies[frequencies] = {1000.0f,  2000.0f,  3000.0f,  4000.0f,  5000.0f,
                                               10000.0f, 20000.0f, 25000.0f, 30000.0f, 35000.0f};
#else
    int frequencies = 1;
    float sampling_frequencies[frequencies] = {35000.0f};
#endif

    for (int i = 0; i < frequencies; i++)
    {
        total_time = 0;

        float sampling_frequency = sampling_frequencies[i];

        int carrier_frequency = 200, frequency_deviation = 50;
        float begin = 0.0f, end = 1.0f, f1 = 5.0f, f2 = 10.0f, f3 = 15.0f;
        float amplitude1 = 1.0f, amplitude2 = 2.0f, amplitude3 = 1.0f;

        thrust::device_vector<float> signal((end - begin) / (1.0f / sampling_frequency) + 1);
        thrust::device_vector<float> time((end - begin) / (1.0f / sampling_frequency) + 1);

        generate_signal(signal, time, begin, end, sampling_frequency, f1, f2, f3, amplitude1, amplitude2, amplitude3);
        thrust::device_vector<float> encoded_signal =
            encode(signal, time, sampling_frequency, carrier_frequency, frequency_deviation);

        total_time = 0;

#ifndef NSYS_COMP
        for (int i = 0; i < 100; i++)
#else
        for (int i = 0; i < 1; i++)
#endif
        {
            thrust::device_vector<float> encoded_signal =
                encode(signal, time, sampling_frequency, carrier_frequency, frequency_deviation);
        }

        std::vector<float> signal_host(signal.size());
        std::vector<float> time_host(time.size());
        std::vector<float> encoded_signal_host(encoded_signal.size());

        thrust::copy(signal.begin(), signal.end(), signal_host.begin());
        thrust::copy(time.begin(), time.end(), time_host.begin());
        thrust::copy(encoded_signal.begin(), encoded_signal.end(), encoded_signal_host.begin());

#ifndef NSYS_COMP
        printf("%f Hz - Average time taken to encode: %f us\n", sampling_frequency, total_time / 100 * 1000);
#else
        printf("%f Hz - Average time taken to encode: %f us\n", sampling_frequency, total_time * 1000);
#endif

        // matplot::figure();
        // matplot::plot(time_host, signal_host);

        // matplot::figure();
        // matplot::plot(time_host, encoded_signal_host);

        // matplot::show();
    }

    return 0;
}