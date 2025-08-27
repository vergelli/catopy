/*
    THIS IS A PALCEHODLER, IGNORE COMPLETELY - TO BE IMPLEMENTED LATER
*/


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gtest/gtest.h>

__global__ void aKernel(int* data) {
    // IMplement here
}

// unit test fixture
class MyKernelTest : public ::testing::Test {
 protected:
    int* d_data;
    int data_size;

    virtual void SetUp() {
        data_size = 1024;
        CUDA_CHECK(cudaMalloc((void**)&d_data, data_size * sizeof(int)));
    }

    virtual void TearDown() {
        CUDA_CHECK(cudaFree(d_data));
    }
};

// CUDA unit test
TEST_F(MyKernelTest, MyKernelWorks) {
    aKernel<<<1, 256>>>(d_data);
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_data;
    CUDA_CHECK(cudaMallocHost((void**)&h_data, data_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(h_data, d_data, data_size * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFreeHost(h_data));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}