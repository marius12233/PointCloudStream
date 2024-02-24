#include "gpu_allocation/gpu_allocation_node.hpp"
#include "gpu_allocation/types_gpu.hpp"
#include "gpu_msgs/PointCloudGPUHandle.h"

GPUAllocationNode::GPUAllocationNode() : m_point_cloud_gpu() {
    //Topic you want to publish

    //std::string pointcloud_filtered_topic{"/pointcloud_topic"};
    //m_publisher_filtered_pointcloud = m_node_handler.advertise<sensor_msgs::PointCloud2>(pointcloud_filtered_topic, 1000);

    std::string pointcloud_gpu_topic{"/pointcloud_gpu_handle"};
    m_publisher_point_cloud_gpu = m_node_handler.advertise<gpu_msgs::PointCloudGPUHandle>(pointcloud_gpu_topic, 1000);
    //Topic you want to subscribe
    std::string pointcloud_topic{"/velodyne_points"};
    m_subscriber_pointcloud = m_node_handler.subscribe(pointcloud_topic, 1000, &GPUAllocationNode::callbackPointCloud, this);


}

std::string toString(cudaIpcMemHandle_t memHandle)
{
	int i;
    int size = 64;
	std::string s = "";
	for (i = 0; i < size; i++) {
		s = s + memHandle.reserved[i];
	}
	return s;
}

void GPUAllocationNode::callbackPointCloud(const PointCloud::Ptr pointcloud) {
    m_point_cloud_gpu.uploadToDevice(*pointcloud);

    std::cout << m_point_cloud_gpu.size() << std::endl;
    cudaIpcMemHandle_t memHandle;
    gpuErrchk(cudaIpcGetMemHandle(&memHandle, m_point_cloud_gpu.devicePointCloudPtr()));

    gpu_msgs::PointCloudGPUHandle msg;
    msg.num_points = m_point_cloud_gpu.size();
    msg.mem_handler = toString(memHandle);

    //m_publisher_filtered_pointcloud.publish(pointcloud);
    m_publisher_point_cloud_gpu.publish(msg);
    
}


int main(int argc, char **argv)
{

  ros::init(argc, argv, "gpu_allocation_node");

  GPUAllocationNode filter_app;

  ros::spin();

  return 0;
}
