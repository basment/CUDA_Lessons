
#include <iostream>
#include <curand_kernel.h>  // CUDA library for random number generation
#include <cuda_runtime.h>   // CUDA runtime API
#include "device_launch_parameters.h" // For device properties, useful for VS

using namespace std;

const int ppt = 2000; //Points Per Thread

__global__ void monteCarloPi(int *count, int ppt,  unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    int localCount = 0;

    //Generate Random Points
    for (int i = 0; i < ppt; i++)
    {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);

        //Check if point is in circle
        if ((x * x + y * y) <= 1.0f)

        {
            localCount++;
        }
    }

    count[idx] = localCount;
}

int main()
{
    int tpb = 256; //Threads Per Block
    int numBlocks = 256; //# of blocks
    int totalThreads = tpb * numBlocks;


    int *d_count;
    cudaMalloc(&d_count, totalThreads * sizeof(int));

    unsigned long long seed = 12345ULL; //Set Seed For # gen

    monteCarloPi << <numBlocks, tpb >> > (d_count, ppt, seed);
    cudaDeviceSynchronize();

    int* h_count = new int[totalThreads];
    cudaMemcpy(h_count, d_count, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

    int totalInCircle = 0;
    for (int i = 0; i < totalThreads; i++)
    {
        totalInCircle += h_count[i];
    }

    int totalPoints = totalThreads * ppt;
    float pi = 4.0f * static_cast<float>(totalInCircle) / static_cast<float>(totalPoints);
    cout << "Approximatied value of Pi: " << pi << endl;
    cout << "Total points: " << totalPoints << endl;

    delete[] h_count;
    cudaFree(d_count);

    return 0;
}
