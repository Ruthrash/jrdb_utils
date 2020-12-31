#include "jrdb_utils/GetCameraModel.h"

GetCameraModel::GetCameraModel(ros::NodeHandle &node)
{
    num_cameras = 10; 
    received_info_flags.resize(num_cameras);
    camerainfo_sub_vec.resize(num_cameras);
    camera_models.resize(num_cameras);
    /*for(int i =0; i < num_cameras ; i++)
    {
        camerainfo_sub_vec.push_back()
    }*/
}