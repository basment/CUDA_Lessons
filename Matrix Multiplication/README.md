# Matrix Multiplication (CPU vs GPU)

This project demonstrates how to implement matrix multiplication on both the CPU and GPU using CUDA, and compares the execution times to highlight the performance benefits of GPU parallelism.

---

## Features
- Implements matrix multiplication for square matrices of size ( N * N ).
- Provides both:
  - **CPU Implementation**: Sequential execution.
  - **GPU Implementation**: Parallel execution using CUDA kernels.
- Measures and compares execution times for CPU and GPU computations.

---

## Prerequisites
To run this project, ensure the following are installed on your system:
- **CUDA Toolkit** 
- **C++ Compiler** 
- **NVIDIA GPU** with CUDA support

---
## Takeaway
This demonstrates the power of pallel processing when used in an effiecent way.

Data: 
  - Each thread is responsible for one element of the result C matrix.
      - Thread actions:
          - Read one row of A
          - Read one column of B
          - Perform N operations
  
  - N^2 = matrix dimension 

When our N vairable is at 100 (Giving us 10,000 elements), you see that the kernel execution is 2x as fast.
- N = 100
    - Cpu execution - 1.6 ms
    - Gpu execution - 0.8 ms (2x faster)
  
The higher you raise the N vairable (The more data that needs to be processed) the larger your gain in effiency.
  - N = 2000
      - Cpu execution - 15,875 ms
      - Gpu execution - 690 ms (23x faster)

However, when N is lower than 80 you begin to see a different result
  - N = 50
      - CPU execution - 0.18 ms
      - GPU execution - 0.95 ms (5x slower) (Because there is such little data to deal with, the CPU is able to process all the data while the GPU copies data to and from the CPU memory prior to execution)

While at times this short (ms) we wouldn't care about these very fractional differences, it is enough to show the difference between CPU and GPU computing.


Ali Chouaib
11/20/2024



