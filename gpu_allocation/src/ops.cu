#include "gpu_allocation/types_gpu.hpp"

// Kernel definition
__global__ void computeRange(const float4* point_cloud, float* ranges, size_t num_current_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if (idx < num_current_points) {
        float range = 0;
        auto coord = point_cloud[idx];
        range += coord.x * coord.x;
        range += coord.y * coord.y;
        range += coord.z * coord.z;
        ranges[idx] = std::sqrt(range);
    }
    
}

void computeRangeKernel(const float4* point_cloud_gpu_raw_data, float* range_array_gpu, size_t num_current_points) {
    int inputSize = num_current_points; // The size of the input data
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    int threadsPerBlock = props.maxThreadsPerBlock;

    int blocksPerGrid = inputSize / threadsPerBlock + ((inputSize % threadsPerBlock) ? 1:0); 
        
    computeRange<<<blocksPerGrid, threadsPerBlock>>>(point_cloud_gpu_raw_data, range_array_gpu, num_current_points);
    gpuErrchk( cudaPeekAtLastError() );
}
