#ifndef REGISTER_RGBD_H
#define REGISTER_RGBD_H

#define M_PI 3.14159265358979323846  /* pi */
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <math.h>

typedef cv::Vec<float, 3> Vec3b;
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
    float f_y = 484.352;//median of calibrated cameras' focal lengths
    float y_0 = 208.799;//nedian of calibrated cameras' offsets
    //float y_0 = 1880;
    ros::Publisher depth_encoded_pub; 
    std::vector<int> GetPixelCordinates(const tf::Point &point);
    int W_img, H_img; 
    cv_bridge::CvImagePtr cv_ptr;
    tf::Transform upper_vel2occam, lower_vel2occam;
};
#endif 