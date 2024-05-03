#include <chrono>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <matplot/matplot.h>
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

__global__ void fill_vector(float *output_signal, float *time, int size, float f1, float f2, float f3)
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
                output_signal[index] =
                    __sinf(f1_precomputed * temp) + __sinf(f2_precomputed * temp) + __sinf(f3_precomputed * temp);
            }
        }
    }
}

__global__ void element_wise_multiplication(float *output, float *input1, float *input2, int size)
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
                output[index] = input1[index] * input2[index];
            }
        }
    }
}

__global__ void element_wise_division(float *output, float scalar, int size)
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
                output[index] = output[index] / scalar;
            }
        }
    }
}

__global__ void generate_encoded_signal(float *output, float *time, float *cumsum, float time_constant,
                                        float cumsum_constant, int size)
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
                output[index] = __cosf(time_constant * time[index] + cumsum_constant * cumsum[index]);
            }
        }
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

    for (int j = idy; j < (size >> 1); j += stride_y)
    {
        for (int i = idx; i < (size >> 1); i += stride_x)
        {
            int index = j * block_size + i;
            if (index < size)
            {
                output[index].x = hilbert_transform_data[index].x * __cosf(exp_const * time[index]) +
                                  hilbert_transform_data[index].y * __sinf(exp_const * time[index]);
                output[index].y = hilbert_transform_data[index].y * __cosf(exp_const * time[index]) -
                                  hilbert_transform_data[index].x * __sinf(exp_const * time[index]);
            }
        }
    }
}

__global__ void phase_calculation_and_unwrapping()
{
}

void generate_signal(thrust::device_vector<float> &signal, thrust::device_vector<float> &time, float begin, float end,
                     float sampling_frequency, float f1, float f2, float f3)
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
                                 signal.size(), f1, f2, f3);
}

thrust::device_vector<float> encode(thrust::device_vector<float> signal, thrust::device_vector<float> time,
                                    float sampling_frequency, float carrier_frequency, float frequency_deviation)
{
    thrust::device_vector<float> cumsum_result(signal.size());
    thrust::device_vector<float> encoded_signal(signal.size());

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
    thrust::inclusive_scan(signal.begin(), signal.end(), cumsum_result.begin());

    dim3 block(block_size, block_size, 1);
    dim3 grid(((signal.size() / 2) + block.x - 1) / block.x, ((signal.size() / 2) + block.y - 1) / block.y);
    element_wise_division<<<grid, block>>>(thrust::raw_pointer_cast(cumsum_result.data()), sampling_frequency,
                                           cumsum_result.size());

    generate_encoded_signal<<<grid, block>>>(
        thrust::raw_pointer_cast(encoded_signal.data()), thrust::raw_pointer_cast(time.data()),
        thrust::raw_pointer_cast(cumsum_result.data()), 2.0f * M_PI * carrier_frequency,
        2.0f * M_PI * frequency_deviation, encoded_signal.size());
	
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	total_time += milliseconds;

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
    int sampling_frequency = 500, carrier_frequency = 20, frequency_deviation = 50;
    float begin = 0.0f, end = 1.0f, f1 = 5.0f, f2 = 10.0f, f3 = 15.0f;
    thrust::device_vector<float> signal((end - begin) / (1.0f / (float)sampling_frequency) + 1);
    thrust::device_vector<float> time((end - begin) / (1.0f / (float)sampling_frequency) + 1);

    generate_signal(signal, time, begin, end, sampling_frequency, f1, f2, f3);

    auto start = std::chrono::high_resolution_clock::now();
    thrust::device_vector<float> encoded_signal =
        encode(signal, time, sampling_frequency, carrier_frequency, frequency_deviation);
    auto stop = std::chrono::high_resolution_clock::now();

	total_time = 0;

    std::cout << "Time taken to encode under cold start: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
              << " us" << std::endl;

	for (int i = 0; i < 100; i++) {
        thrust::device_vector<float> encoded_signal =
            encode(signal, time, sampling_frequency, carrier_frequency, frequency_deviation);
    }

    std::vector<float> signal_host(signal.size());
    std::vector<float> time_host(time.size());
    std::vector<float> encoded_signal_host(encoded_signal.size());

    thrust::copy(signal.begin(), signal.end(), signal_host.begin());
    thrust::copy(time.begin(), time.end(), time_host.begin());
    thrust::copy(encoded_signal.begin(), encoded_signal.end(), encoded_signal_host.begin());

    std::cout << "Average time taken to encode: " << total_time / 100 * 1000 << " us" << std::endl;

    // matplot::figure();
    // matplot::plot(time_host, signal_host);

    // matplot::figure();
    // matplot::plot(time_host, encoded_signal_host);

    // matplot::show();

    return 0;
}