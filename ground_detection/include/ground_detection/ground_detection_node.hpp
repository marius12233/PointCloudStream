#pragma once
#include "ros/ros.h"
#include <sstream>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include "ground_detection/ground_detection.hpp"
#include "gpu_msgs/PointCloudGPUHandle.h"

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
class GroundDetectionNode {
    public:
        GroundDetectionNode();

        void callbackPointCloud(PointCloud::Ptr pointcloud);

        void callbackPointCloudGpu(gpu_msgs::PointCloudGPUHandle point_cloud_gpu);

    private:
        ros::NodeHandle m_node_handler; 
        ros::Publisher m_publisher_filtered_pointcloud;
        ros::Subscriber m_subscriber_point_cloud_gpu;
        ros::Subscriber m_subscriber_pointcloud;

        GroundDetectionGPU<PointCloud> m_ground_detection_gpu;
        PointCloud current_point_cloud;

};