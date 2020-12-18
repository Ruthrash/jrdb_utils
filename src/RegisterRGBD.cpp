#include "jrdb_utils/RegisterRGBD.h"

RegisterRGBD::RegisterRGBD(ros::NodeHandle &node)
{
    upper_velodyne_sub.subscribe(node, "/upper_velodyne/velodyne_points",1);
    lower_velodyne_sub.subscribe(node, "/lower_velodyne/velodyne_points",1); 
    stitched_image_sub.subscribe(node, "/ros_indigosdk_node/stitched_image0/image_raw",1);

    sync_.reset(new Sync(MySyncPolicy(3000), upper_velodyne_sub, lower_velodyne_sub, stitched_image_sub));
    sync_->registerCallback(boost::bind(&FaceRecognition::FaceYOLOSyncCB, dynamic_cast<FaceRecognition*>(this), _1, _2, _3));
}

void RegisterRGBD::SyncVeloRGB(const sensor_msgs::PointCloud2::ConstPtr& upper_velodyne_points,
                                const sensor_msgs::PointCloud2::ConstPtr& lower_velodyne_points,
                                 const sensor_msgs::Image::ConstPtr& stitched_image)
{

}


int main()
{
    return 0; 
}