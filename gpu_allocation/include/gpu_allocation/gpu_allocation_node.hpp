#pragma once
#include "ros/ros.h"
#include <sstream>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include "gpu_allocation/types_gpu.hpp"

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
class GPUAllocationNode {
    public:
        GPUAllocationNode();

        void callbackPointCloud(PointCloud::Ptr pointcloud);

    private:
        ros::NodeHandle m_node_handler; 
        //ros::Publisher m_publisher_filtered_pointcloud;
        ros::Publisher m_publisher_point_cloud_gpu;
        ros::Subscriber m_subscriber_pointcloud;
        PointCloudGPU<PointCloud> m_point_cloud_gpu;

};