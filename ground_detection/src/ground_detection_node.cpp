#include "ground_detection/ground_detection.hpp"
#include "ground_detection/ground_detection_node.hpp"
#include "gpu_allocation/types_gpu.hpp"
#include "gpu_msgs/PointCloudGPUHandle.h"

GroundDetectionNode::GroundDetectionNode() : m_ground_detection_gpu{10000, 0.3f} {
    //Topic you want to publish
    std::string pointcloud_filtered_topic{"/pointcloud_filtered_topic"};
    m_publisher_filtered_pointcloud = m_node_handler.advertise<sensor_msgs::PointCloud2>(pointcloud_filtered_topic, 1000);

    std::string pointcloud_gpu_topic{"/pointcloud_gpu_handle"};
    m_subscriber_point_cloud_gpu = m_node_handler.subscribe<gpu_msgs::PointCloudGPUHandle>(pointcloud_gpu_topic, 1000, &GroundDetectionNode::callbackPointCloudGpu, this);
    //Topic you want to subscribe
    std::string pointcloud_topic{"/velodyne_points"};
    m_subscriber_pointcloud = m_node_handler.subscribe(pointcloud_topic, 1000, &GroundDetectionNode::callbackPointCloud, this);

}


void GroundDetectionNode::callbackPointCloud(const PointCloud::Ptr pointcloud) {

    // Just for test purpose
    current_point_cloud = *pointcloud;
}

cudaIpcMemHandle_t toMemHandler(std::string memHandleStr)
{
	int i;
    int size = 64;
	cudaIpcMemHandle_t mem_handler;
	for (i = 0; i < size; i++) {
		mem_handler.reserved[i] = memHandleStr[i];
	}
    
	return mem_handler;
}

void GroundDetectionNode::callbackPointCloudGpu(gpu_msgs::PointCloudGPUHandle point_cloud_gpu_msg) {
    
    size_t num_points = point_cloud_gpu_msg.num_points;
    std::cout << "Num points in msg: " << num_points << std::endl;
    cudaIpcMemHandle_t mem_handler = toMemHandler(point_cloud_gpu_msg.mem_handler);
    PointCloudGPU<PointCloud> point_cloud_gpu{mem_handler, static_cast<int>(num_points)};
    m_ground_detection_gpu.fit(point_cloud_gpu);

    auto point_cloud = point_cloud_gpu.toHost();
    auto& inliers_mask = m_ground_detection_gpu.getHostInliersMask(point_cloud_gpu.size());

    current_point_cloud.resize(point_cloud_gpu.size());
    int i = 0;
    for(PointCloud::iterator pointcloud_iterator = current_point_cloud.begin();
        pointcloud_iterator!=current_point_cloud.end();
        pointcloud_iterator++){

        if(!static_cast<bool>(inliers_mask[i])) {
            pointcloud_iterator->x = NAN;
            pointcloud_iterator->y = NAN;
            pointcloud_iterator->z = NAN;
        } else {
            pointcloud_iterator->x = point_cloud[i].x;
            pointcloud_iterator->y = point_cloud[i].y;
            pointcloud_iterator->z = point_cloud[i].z;
        }
        i++;
    }

    m_publisher_filtered_pointcloud.publish(current_point_cloud);
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "filter_node");

  GroundDetectionNode filter_app;

  ros::spin();

  return 0;
}
