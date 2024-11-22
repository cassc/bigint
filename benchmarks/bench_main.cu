#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <CuBigInt/bigint.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>
#include <stdexcept>
#include <iostream>

// Define CUDA_CHECK_THROW macro
#define CUDA_CHECK_THROW(call) do {					\
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      std::cerr << "CUDA call failed: " << #call << "\n"                \
                << "Error code: " << err << "\n"                        \
                << "Error string: " << cudaGetErrorString(err)          \
                << " (at " << __FILE__ << ":" << __LINE__ << ")"        \
                << std::endl;                                           \
      throw std::runtime_error(cudaGetErrorString(err));                \
    }                                                                   \
  } while (0)

#ifdef ENABLE_DEBUG
#define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) do {} while (0)
#endif



__global__ void BigIntInitTest_kernel(bigint* a) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx != 0) return;

  bigint c[1];
  bigint_init(c);

  DEBUG_PRINT("BigIntInitTest_kernel multiplication before\n");

  DEBUG_PRINT("Address of a %p\n", a);
  DEBUG_PRINT("Address of a.capacity %d\n", a->capacity);
  DEBUG_PRINT("Address of a+1 %p\n", a + 1);
  DEBUG_PRINT("Address of (a+1).capacity %d\n", (a + 1)->capacity);
  DEBUG_PRINT("Address of (a+2).capacity %d\n", (a + 2)->capacity);

  bigint_mul(c, a, a + 1);

  DEBUG_PRINT("BigIntInitTest_kernel multiplication after\n");

  //NOTE: if we want to return a bigint, we'll need to allocate max size beforehand

  bigint_free(c);
}

static void BM_SimpleMath(benchmark::State& state)
{
  const char *text = "123456790123456790120987654320987654321";
  const char *expected = "15241579027587258039323273891175125743036122542295381801554580094497789971041";
  bigint a[3];
  bigint *device_a;
  for (int i = 0; i < 3; i++) bigint_init(a + i);

  bigint_from_str_base(a, text, 10);
  bigint_from_str_base(a + 1, text, 10);
  bigint_from_str_base(a + 2, expected, 10);

  DEBUG_PRINT("CudaTestBigIntFromStrBase: sizeof(a) %ld\n", sizeof(a));

  CUDA_CHECK_THROW(cudaMalloc(&device_a, sizeof(a)));
  CUDA_CHECK_THROW(cudaMemcpy(device_a, a, sizeof(a), cudaMemcpyHostToDevice));

  for (int i = 0; i < 3; i++) {
    // Allocate memory for words on the device
    bigint_word *device_words;
    DEBUG_PRINT("CudaTestBigIntFromStrBase: a[%d].capacity %d\n", i, a[i].capacity);
    cudaMalloc(&device_words, a[i].capacity * sizeof(bigint_word));

    // Copy words data from host to device
    cudaMemcpy(device_words, a[i].words, a[i].size * sizeof(bigint_word), cudaMemcpyHostToDevice);

    // Update the device `bigint` structure's `words` pointer
    DEBUG_PRINT("Updating a[%d].words to %p\n", i, device_words);
    cudaMemcpy(&device_a[i].words, &device_words, sizeof(bigint_word *), cudaMemcpyHostToDevice);

    // Copy other fields explicitly
    cudaMemcpy(&device_a[i].neg, &a[i].neg, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_a[i].size, &a[i].size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_a[i].capacity, &a[i].capacity, sizeof(int), cudaMemcpyHostToDevice);
  }


  CUDA_CHECK_THROW(cudaDeviceSynchronize());

  for (auto _: state) {
    BigIntInitTest_kernel<<<1, 1>>>(device_a);
  }

  CUDA_CHECK_THROW(cudaDeviceSynchronize());

  // todo how to free the two pointers in the device?
  CUDA_CHECK_THROW(cudaFree(device_a));
}

BENCHMARK(BM_SimpleMath);



BENCHMARK_MAIN();
