#pragma once

#include <cuda_runtime.h>
#include <array>
#include "gpu_allocation/types_gpu.hpp"

void groundPointsDetection(const float4* d_point_cloud, size_t num_current_points, int* d_inliers_mask, float4* d_fitting_plane, int* num_inliers, float distance_threshold, int max_iterations);

class GroundDetectionGPU {

    public:

        GroundDetectionGPU(int max_iterations_, float distance_threshold_) : 
            max_iterations(max_iterations_), 
            distance_threshold(max_iterations){
            
            onDeviceAlloc();

        }

        void fit(const PointCloudGPU& point_cloud_gpu) {

            size_t num_current_points = point_cloud_gpu.size();
            const float4* d_point_cloud = point_cloud_gpu.devicePointCloudPtr();

            groundPointsDetection(d_point_cloud, num_current_points, d_inliers_mask, d_fitting_plane, d_num_inliers, distance_threshold, max_iterations);
        }

        int* deviceInliersMask() {return d_inliers_mask;};
        float4* deviceFittingPlane() {return d_fitting_plane;};
        int* deviceNumInliers() { return d_num_inliers; };


    
    private:

        void onDeviceAlloc() {
            cudaMalloc(&d_inliers_mask, MAX_NUM_POINTS * sizeof(int));
            cudaMalloc(&d_fitting_plane, sizeof(float4));
            cudaMalloc(&d_num_inliers, sizeof(int));
            cudaMemset(&d_num_inliers, 0, sizeof(int));
        };

        int max_iterations{1000};
        float distance_threshold{0.1f};
        // Device variables
        int* d_inliers_mask;
        float4* d_fitting_plane;
        int* d_num_inliers;

};
