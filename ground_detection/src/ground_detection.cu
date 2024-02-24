#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <curand_kernel.h>
#include <random>
#include "ground_detection/ground_detection.hpp"

#define BLOCK_DIM 128

__device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float distance(const float3& p, const float3& n, float d) {
    return fabs(dot(p, n) + d) / sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
}

__device__ void atomicUpdate(int* address, int val_to_compare_and_store, float4* float4_address, float4 val_to_store)
{

    int ret = *address;

    while(val_to_compare_and_store > ret) {

        int old = ret;
        ret = atomicCAS(address, old, val_to_compare_and_store);
        if(ret == old) { // it means that the current thread has modified the value and returned the old value
            *float4_address = val_to_store;
            break;
        }
    }
}

__global__ void ransac(const float4*  pointCloud, int numPoints, float4* bestPlane, int* inliers, float distanceThreshold, int maxIterations) {
    
    __shared__ float s_bestPlane[BLOCK_DIM]; // Shared memory for storing each thread's best plane parameters
    s_bestPlane[threadIdx.x] = 0; // Initialize the number of inliers for the current thread
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx >= maxIterations) return;

    curandState state;
    curand_init((unsigned long long)clock() + idx, 0, 0, &state);

    int index[3];
    for (int i = 0; i < 3; ++i)
        index[i] = curand(&state) % numPoints;

    float3 p1 = make_float3( pointCloud[index[0]].x,  pointCloud[index[0]].y,  pointCloud[index[0]].z);
    float3 p2 = make_float3( pointCloud[index[1]].x,  pointCloud[index[1]].y,  pointCloud[index[1]].z);
    float3 p3 = make_float3( pointCloud[index[2]].x,  pointCloud[index[2]].y,  pointCloud[index[2]].z);

    float3 normal = cross(p2 - p1, p3 - p1);
    float length = sqrtf(dot(normal, normal));
    normal = normal / length;

    float d = -dot(normal, p1);

    int numInliers = 0;

    for (int i = 0; i < numPoints; ++i) {
        auto& point = pointCloud[i];
        float dist = distance(make_float3( point.x,  point.y,  point.z), normal, d);
        if (dist <= distanceThreshold)
            numInliers++;
    }

    s_bestPlane[threadIdx.x] = numInliers; // Store the number of inliers for this thread in shared memory

    __syncthreads(); // Ensure all threads have stored their number of inliers before reduction

    // Reduction to find the best solution among all threads of the same block
    // This solution allows to reduce the number of threads trying to update values on memory.
    if (threadIdx.x == 0) {
        double maxInliers = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            if (s_bestPlane[i] > maxInliers) {
                maxInliers = s_bestPlane[i];
            }
        }
        float4 current_plane = make_float4(normal.x, normal.y, normal.z, d);
        atomicUpdate(&inliers[0], numInliers, bestPlane, current_plane);
    }
}

__global__ void getInliersMask(const float4* pointCloud, int numPoints, const float4* bestPlane, int* inliersMask, float distanceThreshold) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPoints) return;

    inliersMask[idx] = 0; // pre-set all the values of the inliersMask to 0.

    auto& point = pointCloud[idx];
    float3 normal = make_float3(bestPlane->x, bestPlane->y, bestPlane->z);
    float d = bestPlane->w;
    float dist = distance(make_float3( point.x, point.y, point.z), normal, d);
    if (dist <= distanceThreshold)
        inliersMask[idx] = 1;

}

void groundPointsDetection(const float4* d_point_cloud, size_t num_current_points, int* d_inliers_mask, float4* d_fitting_plane, int* num_inliers, float distance_threshold, int max_iterations) {
    int inputSize = num_current_points; // The size of the input data
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    int threadsPerBlock = BLOCK_DIM;

    int blocksPerGrid = inputSize / threadsPerBlock + ((inputSize % threadsPerBlock) ? 1:0); 
    
    printf("# blocks: %d # threads: %d", blocksPerGrid, threadsPerBlock);
    ransac<<<blocksPerGrid, threadsPerBlock>>>(d_point_cloud, num_current_points, d_fitting_plane, num_inliers, distance_threshold, max_iterations);
    gpuErrchk( cudaPeekAtLastError() );

    getInliersMask<<<blocksPerGrid, threadsPerBlock>>>(d_point_cloud, num_current_points, d_fitting_plane, d_inliers_mask, distance_threshold);
    gpuErrchk( cudaPeekAtLastError() );
}

/*
// For test purpose
// TODO: Use in unit tests
std::vector<float4> generatePointCloudOnPlane(int numPoints, double planeHeight) {
    std::vector<float4> points;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-10.0, 10.0); // Adjust range as needed
    std::uniform_real_distribution<double> dis_z(-0.05, 0.05); // Adjust range as needed

    for (int i = 0; i < numPoints; ++i) {
        float4 p;
        p.x = dis(gen);
        p.y = dis(gen);
        p.z = planeHeight + dis_z(gen); // All points are on the same plane parallel to XY plane
        points.push_back(p);
    }

    return points;
}

int main() {
    std::vector<float4> points = generatePointCloudOnPlane(100000, 0);//{{1, 2, 0.1}, {4, 5, 0.05}, {7, 8, 0.01}, {10, 11, 10.002}, {13, 14, 10.015}};
    int numPoints = points.size();

    float distanceThreshold = 0.1f;
    int maxIterations = 10000;
    
    float4* d_points;
    float4* d_bestPlane;
    int* d_inliers;

    cudaMalloc((void**)&d_points, numPoints * sizeof(float4));
    cudaMemcpy(d_points, points.data(), numPoints * sizeof(float4), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_bestPlane, sizeof(float4)); // One for normal, one for d
    cudaMalloc((void**)&d_inliers, sizeof(int));

    int* inliers = new int;
    *inliers = 0;

    cudaMemcpy(d_inliers, inliers, sizeof(int), cudaMemcpyHostToDevice);

    int* d_inliers_mask;
    cudaMalloc((void**)&d_inliers_mask, numPoints * sizeof(int));

    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid = (maxIterations + threadsPerBlock - 1) / threadsPerBlock;

    // Use APIs for thread occupancy
    //===============================
    // int numBlocks;        // Occupancy in terms of active blocks
    // int blockSize = 1024;

    // // These variables are used to convert occupancy to warps
    // int device;
    // cudaDeviceProp prop;
    // int activeWarps;
    // int maxWarps;

    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&prop, device);

    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks,
    //     ransac,
    //     blockSize,
    //     0);

    // activeWarps = numBlocks * blockSize / prop.warpSize;
    // maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    // std::cout << "Max active blocks: " << numBlocks << std::endl;
    // std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
    // std::cout << "Max warps:: " << maxWarps << std::endl;
    //===============================
    ///

    ransac<<<blocksPerGrid, threadsPerBlock>>>(d_points, numPoints, d_bestPlane, d_inliers, distanceThreshold, maxIterations);
    gpuErrchk( cudaPeekAtLastError() );

    threadsPerBlock = BLOCK_DIM;
    blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    getInliersMask<<<blocksPerGrid, threadsPerBlock>>>(d_points, numPoints, d_bestPlane, d_inliers_mask, distanceThreshold);
    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(inliers, d_inliers, sizeof(int), cudaMemcpyDeviceToHost);

    float4 bestPlane[1];
    cudaMemcpy(bestPlane, d_bestPlane, sizeof(float4), cudaMemcpyDeviceToHost);

    int inliersMask[numPoints];
    cudaMemcpy(inliersMask, d_inliers_mask, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Best plane parameters: Normal = (" << bestPlane[0].x << ", " << bestPlane[0].y << ", " << bestPlane[0].z << "), d = " << bestPlane[0].w << std::endl;

    std::cout << "inliers: " << inliers[0] << std::endl;

    bool isWorking = true;
    for(int i=0; i<numPoints; i++) {
        if(inliersMask[i] != 1) {
            isWorking=false;
            std::cout << "Not working at idx: " << i << std::endl;
        };
    }

    std::cout << "is working: " << isWorking << std::endl;
    
    cudaFree(d_points);
    cudaFree(d_bestPlane);
    cudaFree(d_inliers);
    delete inliers;

    return 0;
}
*/
