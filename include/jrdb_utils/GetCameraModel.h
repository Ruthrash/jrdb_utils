#ifndef GET_CAMERA_MODEL_H
#define GET_CAMERA_MODEL_H

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <vector>
#include <image_geometry/pinhole_camera_model.h>
class GetCameraModel
{

public:
    std::vector<bool> received_info_flags;
    GetCameraModel(){}
    ~GetCameraModel(){}
    GetCameraModel(ros::NodeHandle &node);

protected:
    

private:
    int num_cameras;
    std::vector<ros::Subscriber> camerainfo_sub_vec;
    std::vector<image_geometry::PinholeCameraModel> camera_models;
    image_geometry::PinholeCameraModel final_camera_model;

};


#endif