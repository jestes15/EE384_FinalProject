#include <cuda_runtime.h>
#include <matplot/matplot.h>
#include <stdio.h>

void unwrap_phase(float *h_input, float *h_output, int N)
{
    for (int i = 0; i < N; ++i)
    {
        if (i == 0)
        {
            h_output[i] = h_input[i];
        }
        else
        {
            float diff = h_input[i] - h_input[i - 1];
            if (diff > M_PI)
            {
                h_output[i] += 2 * M_PI;
            }
            else
            {
                h_output[i] = h_input[i];
            }
        }
    }
}

int main()
{
    const int N = 20;
    const float sampling_frequency = 3.0f; // Example sampling frequency

    std::vector<float> time(N);
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);

    float counter = 0;
    for (int i = 0; i < N; ++i)
    {
        time[i] = i;

        if (counter > M_PI)
        {
            counter = -M_PI;
        }

        h_input[i] = counter;
        counter += M_PI * (1 / sampling_frequency);
    }

    unwrap_phase(h_input.data(), h_output.data(), N);

    matplot::plot(time, h_input);
    matplot::hold(matplot::on);
    matplot::plot(time, h_output);
    matplot::hold(matplot::off);
    matplot::show();

    return 0;
}