#pragma once

#include <cuda_runtime.h>
#include <array>

static constexpr std::size_t MAX_NUM_POINTS = 100000;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void computeRangeKernel(const float4* point_cloud_gpu_raw_data, float* ranges, size_t num_current_points);

template<typename PointCloudT>
class PointCloudGPU {
    public:

        PointCloudGPU() {
            const size_t bytes = MAX_NUM_POINTS * sizeof(float4);
            gpuErrchk(cudaMalloc(&d_point_cloud_ptr, bytes));
        };

        
        PointCloudGPU(cudaIpcMemHandle_t mem_handler, int num_points) {
            m_use_mem_handler = true;
            gpuErrchk(cudaIpcOpenMemHandle((void **) &d_point_cloud_ptr, mem_handler,
                            cudaIpcMemLazyEnablePeerAccess));
        };
        
        void uploadToDevice(PointCloudT& point_cloud) {
            
            int idx_field = 0;
            for(const auto &point : point_cloud) {
                point_cloud_array[idx_field++] = float4{point.x, point.y, point.z, point.intensity}; 
            }

            num_points = point_cloud.size();
            const size_t bytes = num_points * sizeof(float4);
            gpuErrchk(cudaMemcpy(d_point_cloud_ptr, point_cloud_array.data(), bytes, cudaMemcpyHostToDevice));            
        };

        PointCloudGPU(PointCloudT& point_cloud) {
            num_points = point_cloud.size();
            const size_t bytes = num_points * sizeof(float4);

            gpuErrchk(cudaMalloc(&d_point_cloud_ptr, bytes));

            int idx_field = 0;
            for(const auto &point : point_cloud) {
                point_cloud_array[idx_field++] = float4{point.x, point.y, point.z, point.intensity}; 
            }

            gpuErrchk(cudaMemcpy(d_point_cloud_ptr, point_cloud_array.data(), bytes, cudaMemcpyHostToDevice));
        };

        ~PointCloudGPU() {
            
            if(m_use_mem_handler){
                gpuErrchk(cudaIpcCloseMemHandle(d_point_cloud_ptr));
            } else {gpuErrchk(cudaFree(d_point_cloud_ptr));}

        };

        void computeRange(std::array<float, MAX_NUM_POINTS>& range_array) {
            float* d_ranges_ptr;
            const size_t bytes = num_points * sizeof(float);
            gpuErrchk(cudaMalloc(&d_ranges_ptr, bytes));

            computeRangeKernel(d_point_cloud_ptr, d_ranges_ptr, num_points); //point_cloud_array.size());

            gpuErrchk(cudaMemcpy(range_array.data(), d_ranges_ptr, bytes, cudaMemcpyDeviceToHost));
            gpuErrchk(cudaFree(d_ranges_ptr));

        };
        //std::array<float4, MAX_NUM_POINTS>& toHost();

        size_t size() { return num_points; };

        float4* devicePointCloudPtr() {return d_point_cloud_ptr;};
    
    private:
        std::array<float4, MAX_NUM_POINTS> point_cloud_array;
        float4 *d_point_cloud_ptr;
        size_t num_points{0};
        bool m_use_mem_handler{false};

};



