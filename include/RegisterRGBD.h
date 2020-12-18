#ifndef REGISTER_RGBD_H
#define REGISTER_RGBD_H

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
class RegisterRGBD
{
public:
    RegisterRGBD(){}
    ~RegisterRGBD(){}
    RegisterRGBD(ros::NodeHandle &node);
protected:

private:
    void SyncVeloRGB(const sensor_msgs::PointCloud2::ConstPtr& upper_velodyne_points, const sensor_msgs::PointCloud2::ConstPtr& lower_velodyne_points , const sensor_msgs::Image::ConstPtr& stitched_image);
    
    message_filters::Subscriber<sensor_msgs::PointCloud2>upper_velodyne_sub, lower_velodyne_sub;
    message_filters::Subscriber<sensor_msgs::Image> stitched_image_sub; 

    typedef message_filters::sync_policies::ApproximateTime <sensor_msgs::PointCloud2, sensor_msgs::PointCloud2 ,sensor_msgs::Image> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;

};
#endif 