
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int size)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0;
	for (int k = 0; k < size; k++)
	{
		sum += A[row * size + k] * B[k * size + col];
	}
	C[row * size + col] = sum;
}


void matrixMulCPU(const float* A, const float* B, float* C, int size, bool fullOutput = true)
{
	if (!fullOutput && size > 10) 
	{ // For large matrices, only print a subset
		std::cout << "Matrix too large to display.\n";
		return;
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < size; k++)
			{
				sum += A[i * size + k] * B[k * size + j];
			}
			C[i * size + j] = sum;
		}
	}
}


void printMatrix(const float* matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			std::cout << matrix[i * size + j] << " ";
		}
		std::cout << "\n";
	}
}

int main() 
{
	//CPU Matrix Mul
	float* h_a = new float[N * N]; // Matrix A
	float* h_b = new float[N * N]; // Matrix B
	float* h_c = new float[N * N]; // Matrix C (Result)


	for (int i = 0; i < N * N; i++)
	{
		h_a[i] = static_cast<float>(i + 1);
		h_b[i] = static_cast<float>(N * N - i);

	}

	//GPU Matrix Mul
	float* d_a;
	float* d_b;
	float* d_c;

	cudaMalloc((void**)&d_a, N * N * sizeof(float));
	cudaMalloc((void**)&d_b, N * N * sizeof(float));
	cudaMalloc((void**)&d_c, N * N * sizeof(float));

	cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);


	dim3 blockDim(32, 32);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

	
	//Calculate CPU Execution Time
	auto start = std::chrono::high_resolution_clock::now();
	matrixMulCPU(h_a, h_b, h_c, N);
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> cpuDuration = stop - start;
	std::cout << "\nCpu Excecution Time: " << cpuDuration.count() << "ms\n";


	// Calculate GPU execution time
	cudaEvent_t gpuStart, gpuStop;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuStop);

	cudaEventRecord(gpuStart);
	matrixMulKernel << <gridDim, blockDim >> > (d_a, d_b, d_c, N);
	cudaEventRecord(gpuStop);
	cudaEventSynchronize(gpuStop);

	float gpuDuration;
	cudaEventElapsedTime(&gpuDuration, gpuStart, gpuStop);

	cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "\nGPU Execution Time: " << gpuDuration << " ms\n";

	delete[] h_a;
	delete[] h_b;
	delete[] h_c;

	cudaEventDestroy(gpuStart);
	cudaEventDestroy(gpuStop);


	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


}

