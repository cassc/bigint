#include <gtest/gtest.h>
#include <CuBigInt/bigint.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>
#include <stdexcept>
#include <iostream>

// Define CUDA_CHECK_THROW macro
#define CUDA_CHECK_THROW(call) do {                                   \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        std::cerr << "CUDA call failed: " << #call << "\n"            \
                  << "Error code: " << err << "\n"                    \
                  << "Error string: " << cudaGetErrorString(err)      \
                  << " (at " << __FILE__ << ":" << __LINE__ << ")"    \
                  << std::endl;                                       \
        throw std::runtime_error(cudaGetErrorString(err));            \
    }                                                                 \
} while (0)

#ifdef ENABLE_DEBUG
    #define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...) do {} while (0)
#endif


class CudaBigIntTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
        cudaMalloc(&d_result, sizeof(int32_t));
    }

    void TearDown() override {
        cudaFree(d_result);
        cudaDeviceReset();
    }

    int32_t *d_result;
};

__global__ void RandomizedOperations_kernel(int32_t *result) {
    curandState localState;
    curand_init(1234, 0, 0, &localState);
    bigint a[1], b[1], c[1], d[1], e[20];
    bigint_init(a);
    bigint_init(b);
    bigint_init(c);
    bigint_init(d);
    for (int i = 0; i < 20; i++) bigint_init(e + i);
    *result = 0;
    for (int i = 0; i < 12345; i++) {
        int x = curand(&localState) % 12345;
        int y = curand(&localState) % 12345;
        int shift = curand(&localState) % 1234;
        if (curand(&localState) & 1) x = -x;
        if (curand(&localState) & 1) y = -y;

        bigint_from_int(a, x);
        bigint_from_int(b, y);
        bigint_from_int(e + 0, x + y);
        bigint_from_int(e + 1, x - y);
        bigint_from_int(e + 2, x * y);

        if (y != 0) {
            bigint_from_int(e + 3, x / y);
            bigint_from_int(e + 4, x % y);
        }

        bigint_from_int(e + 5, x);
        bigint_from_int(e + 6, bigint_int_gcd(x, y));

        bigint_cpy(c, a);
        bigint_shift_left(a, a, shift);
        bigint_shift_right(a, a, shift);

        *result |= bigint_cmp(a, c);

        bigint_add(e + 10, a, b);
        bigint_sub(e + 11, a, b);
        bigint_mul(e + 12, a, b);
        bigint_div(e + 13, a, b);
        bigint_mod(e + 14, a, b);
        bigint_from_int(e + 15, x);
        bigint_gcd(e + 16, a, b);

        for (int j = 0; j < 7; j++) {
            if (y == 0 && (j == 3 || j == 4)) continue;
            if (bigint_cmp(e + j, e + j + 10) != 0) {
                DEBUG_PRINT("i %i, j %i failed for bigints %i, %i\n", i, j, x, y);
            }
            *result |= bigint_cmp(e + j, e + j + 10);
        }
    }
    bigint_free(a);
    bigint_free(b);
    bigint_free(c);
    bigint_free(d);
    for (int i = 0; i < 20; i++) bigint_free(e + i);
}


TEST_F(CudaBigIntTest, RandomizedOperations) {
    int32_t result = 8;
    RandomizedOperations_kernel<<<8, 8>>>(d_result);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaMemcpy(&result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(result, 0);
}

__global__ void BigIntInitTest_kernel(bigint* a) {
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

  int r = bigint_cmp(a + 2, c);

  if (r != 0) {
    DEBUG_PRINT("BigIntInitTest_kernel failed\n");
  }

  if (r) {
    DEBUG_PRINT("BigIntInitTest_kernel computation failed\n");
    assert(false);
  }

  bigint_free(c);
}

TEST_F(CudaBigIntTest, CudaTestBigIntFromStrBase) {
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

    BigIntInitTest_kernel<<<1, 1>>>(device_a);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    // todo how to free the two pointers in the device?
    CUDA_CHECK_THROW(cudaFree(device_a));
}
