#include "gpu_allocation/gpu_allocation_node.hpp"
#include "gpu_allocation/ground_detection.hpp"
#include "gpu_allocation/types_gpu.hpp"

GPUAllocationNode::GPUAllocationNode() : m_ground_detection_gpu{100000, 0.15f} {
    //Topic you want to publish
    std::string pointcloud_filtered_topic{"/pointcloud_filtered_topic"};
    m_publisher_filtered_pointcloud = m_node_handler.advertise<sensor_msgs::PointCloud2>(pointcloud_filtered_topic, 1000);
    //Topic you want to subscribe
    std::string pointcloud_topic{"/velodyne_points"};
    m_subscriber_pointcloud = m_node_handler.subscribe(pointcloud_topic, 1000, &GPUAllocationNode::callbackPointCloud, this);


}

void GPUAllocationNode::callbackPointCloud(const PointCloud::Ptr pointcloud) {
    PointCloudGPU<PointCloud> point_cloud_gpu(*pointcloud);
    m_ground_detection_gpu.fit(point_cloud_gpu);

    auto& inliers_mask = m_ground_detection_gpu.getHostInliersMask(point_cloud_gpu.size());
    int i = 0;
    for(PointCloud::iterator pointcloud_iterator = pointcloud->begin();
        pointcloud_iterator!=pointcloud->end();
        pointcloud_iterator++){

        if(!static_cast<bool>(inliers_mask[i])) {
            pointcloud_iterator->x = NAN;
            pointcloud_iterator->y = NAN;
            pointcloud_iterator->z = NAN;
        }
        i++;
    }
    m_publisher_filtered_pointcloud.publish(pointcloud);
}


int main(int argc, char **argv)
{

  ros::init(argc, argv, "filter_node");

  GPUAllocationNode filter_app;

  ros::spin();

  return 0;
}
