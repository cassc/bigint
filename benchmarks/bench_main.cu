#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

// CUDA kernel for benchmarking
__global__ void cuda_add(int *a, int *b, int *c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// Benchmark function
static void BM_CUDA_Add(benchmark::State &state) {
    const int size = state.range(0);
    int *a, *b, *c, *d_a, *d_b, *d_c;

    // Allocate host memory
    a = (int *)malloc(size * sizeof(int));
    b = (int *)malloc(size * sizeof(int));
    c = (int *)malloc(size * sizeof(int));

    // Initialize host data
    for (int i = 0; i < size; ++i) {
        a[i] = i;
        b[i] = i;
    }

    // Allocate device memory
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Benchmark loop
    for (auto _ : state) {
        cuda_add<<<(size + 255) / 256, 256>>>(d_a, d_b, d_c, size);
        cudaDeviceSynchronize();
    }

    // Cleanup
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
}

// Register benchmark and input sizes
BENCHMARK(BM_CUDA_Add)->Arg(1024)->Arg(1024 * 1024);

BENCHMARK_MAIN();
