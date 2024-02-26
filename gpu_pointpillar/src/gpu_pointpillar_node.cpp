/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "ros/ros.h"
#include <sstream>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <unistd.h>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "pointpillar.hpp"
#include "common/check.hpp"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2/LinearMath/Quaternion.h>
//#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


using namespace std::chrono_literals;
typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;


/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

std::shared_ptr<pointpillar::lidar::Core> create_core() {
    pointpillar::lidar::VoxelizationParameter vp;
    vp.min_range = nvtype::Float3(0.0, -39.68f, -3.0);
    vp.max_range = nvtype::Float3(69.12f, 39.68f, 1.0);
    vp.voxel_size = nvtype::Float3(0.16f, 0.16f, 4.0f);
    vp.grid_size =
        vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
    vp.max_voxels = 40000;
    vp.max_points_per_voxel = 32;
    vp.max_points = 300000;
    vp.num_feature = 4;

    pointpillar::lidar::PostProcessParameter pp;
    pp.min_range = vp.min_range;
    pp.max_range = vp.max_range;
    pp.feature_size = nvtype::Int2(vp.grid_size.x/2, vp.grid_size.y/2);

    pointpillar::lidar::CoreParameter param;
    param.voxelization = vp;
    param.lidar_model = "/home/cem2brg/point_cloud_stream/src/gpu_pointpillar/model/pointpillar.plan";
    param.lidar_post = pp;
    return pointpillar::lidar::create_core(param);
}

int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: "<< file<<std::endl;
        return -1;
    }

    unsigned int len = 0;
    dataFile.seekg (0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg (0, dataFile.beg);

    char *buffer = new char[len];
    if (buffer==NULL) {
        std::cout << "Can't malloc buffer."<<std::endl;
        dataFile.close();
        exit(EXIT_FAILURE);
    }

    dataFile.read(buffer, len);
    dataFile.close();

    *data = (void*)buffer;
    *length = len;
    return 0;  
}

class MinimalPublisher
{
public:
  MinimalPublisher()
  { 
    
    bool timer = false;
    core = create_core();
    if (core == nullptr) {
        printf("Core has been failed.\n");
    }

    
    cudaStreamCreate(&stream);
  
    core->print();
    core->set_timer(timer);


    //publisher_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("bbox", 700);
    m_subscriber_pointcloud = m_node_handler.subscribe("/velodyne_points", 1000, &MinimalPublisher::topic_callback, this);
    marker_pub = m_node_handler.advertise<visualization_msgs::MarkerArray>("visualization_marker", 1);
  }

private:
  std::vector<std::string> class_names;
  float nms_iou_thresh;
  int pre_nms_top_n;
  bool do_profile{false};
  std::string model_path;
  std::string engine_path;
  std::string data_type;
  float intensity_scale;
  //tf2::Quaternion myQuaternion;
  cudaStream_t stream = NULL;
  ros::NodeHandle m_node_handler; 
  ros::Subscriber m_subscriber_pointcloud;
  ros::Publisher marker_pub;

  std::shared_ptr<pointpillar::lidar::Core> core;


  void topic_callback(const PointCloud::Ptr pcl_cloud)
  { 

    //std::vector<Bndbox> nms_pred;
    //nms_pred.reserve(100);

    unsigned int points_size = pcl_cloud->points.size();

    std::vector<float> pcl_data;

    for (const auto& point : pcl_cloud->points) {
      pcl_data.push_back(point.x);
      pcl_data.push_back(point.y);
      pcl_data.push_back(point.z);
      pcl_data.push_back(point.intensity / 255.);
    }

    std::cout << "Intensity 1st point: " << pcl_data[3] << std::endl;
    float* points = static_cast<float *>(pcl_data.data());
    
    
    //Use 4 because PCL has padding (4th value now has intensity information)
    //unsigned int points_data_size = points_size * sizeof(float) * 4;

    std::cout << "Lidar points count: "<< points_size <<std::endl;
    
    auto bboxes = core->forward(points, points_size, stream);
    std::cout<<"Detections after NMS: "<< bboxes.size()<<std::endl;


    //auto pc_detection_arr = std::make_shared<vision_msgs::msg::Detection3DArray>();

    
    visualization_msgs::MarkerArray markerArray;

    for(int i=0; i<bboxes.size(); i++) {
      //std::cout << "Class: " << std::to_string(nms_pred[i].id) << std::endl;
      //std::cout << "Score: " << nms_pred[i].score << std::endl;
      if(bboxes[i].id == 0 && bboxes[i].score > 0.2) {
        auto& pred = bboxes[i];
        uint32_t shape = visualization_msgs::Marker::CUBE;
        visualization_msgs::Marker marker;
        marker.header.frame_id = "velodyne";
        marker.header.stamp = ros::Time::now();
        marker.id = i;

        marker.type = shape;
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(1);;
        marker.pose.position.x = pred.x;
        marker.pose.position.y = pred.y;
        marker.pose.position.z = pred.z;

        tf2::Quaternion myQuaternion;
        myQuaternion.setRPY( 0, 0, pred.rt - 1.56);

        marker.pose.orientation.x = myQuaternion.getX();
        marker.pose.orientation.y = myQuaternion.getY();
        marker.pose.orientation.z = myQuaternion.getZ();
        marker.pose.orientation.w = myQuaternion.getW();

        marker.scale.x = pred.l;
        marker.scale.y = pred.w;
        marker.scale.z = pred.h;

        marker.color.r = 0.0f;
        marker.color.g = 0.0f;
        marker.color.b = 1.0f;
        marker.color.a = 0.5;
        markerArray.markers.push_back(marker);
      }
    }


    marker_pub.publish(markerArray);


  
  }

  size_t count_;
};

int main(int argc, char * argv[])
{

  ros::init(argc, argv, "pp_infer");

  MinimalPublisher filter_app;

  ros::spin();
  return 0;
}
