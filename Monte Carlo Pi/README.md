# Monte Carlo Simulation for Approximating Pi 

This project demonstrates how to implement a Monte Carlo simulation on the GPU using CUDA to approximate the value of π (Pi). The project leverages GPU parallelism to efficiently simulate a large number of random points and determine how many fall within a quarter circle.

---

## Features
- Implements a Monte Carlo simulation to approximate π using CUDA.
- Utilizes GPU parallelism to speed up random point generation and calculation.
- Uses CUDA's `curand` library for random number generation.
- Measures the number of points that fall inside a quarter circle to estimate π.

---

## Prerequisites
To run this project, ensure the following are installed on your system:
- **CUDA Toolkit**
- **C++ Compiler**
- **NVIDIA GPU** with CUDA support

---

## Takeaway


#### Data and Computation
- **Points Generation**: Each thread is responsible for generating multiple random points and checking if they fall within the quarter circle.
  - Thread actions:
    - Generate random `x` and `y` coordinates.
    - Check if the point lies within a radius of 1 to determine if it falls inside the circle.
- The value of π is estimated as:

  π ≈ 4 × (Number of points inside the circle / Total number of points)

#### Performance Insights
- **Parallelism Gains**: The GPU can generate a large number of points simultaneously, significantly speeding up the calculation compared to sequential CPU execution.
- **Scalability**: The larger the number of points generated, the more accurate the approximation becomes, and the greater the benefit of parallel execution.

---

## Example Results
- **Total Points**: 65,536,000 (256 blocks × 256 threads per block × 1000 points per thread)
  - **Approximated Value of Pi**: Typically close to 3.14159.

The more points generated, the more accurate the estimation of π. GPU parallelism ensures that a larger number of points can be generated efficiently, providing a more precise approximation.

---

Ali Chouaib  
11/25/2024
