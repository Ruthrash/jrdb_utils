#include "jrdb_utils/RegisterRGBD.h"

RegisterRGBD::RegisterRGBD(ros::NodeHandle &node)
{
    tf::Quaternion m_q;
    tf::Transform upper_vel2base, lower_vel2base, occam2base; 

    upper_vel2base.setOrigin( tf::Vector3(-0.019685, 0, 1.077382) );
    m_q.setRPY(0, 0, 0.085);
    //upper_vel2base.setOrigin( tf::Vector3(0, 0, -0.33529) );
    //m_q.setRPY(0, 0, 0.085);
    
    upper_vel2base.setRotation(m_q);

    lower_vel2base.setOrigin( tf::Vector3(-0.019685,0, 0.606982) );
    m_q.setRPY(0,0,0);
    
    //lower_vel2base.setOrigin( tf::Vector3(0, 0, 0.13511));
    //m_q.setRPY(0,0,0);
    
    lower_vel2base.setRotation(m_q);

    //upper_vel2occam = upper_vel2base;
    //lower_vel2occam = lower_vel2base;
   
    occam2base.setOrigin( tf::Vector3(-0.019685, 0.0, 0.742092 ));
    m_q.setRPY(-M_PI/2, 0.0, -M_PI/2);
    occam2base.setRotation(m_q);

    upper_vel2occam = occam2base.inverse()*upper_vel2base, lower_vel2occam = occam2base.inverse()*lower_vel2base;
    

    upper_velodyne_sub.subscribe(node, "/upper_velodyne/velodyne_points",100);
    lower_velodyne_sub.subscribe(node, "/lower_velodyne/velodyne_points",100); 
    stitched_image_sub.subscribe(node, "/ros_indigosdk_node/stitched_image0/image_raw",100);
    depth_encoded_pub = node.advertise<sensor_msgs::Image>("/depth_modified_pub", 1);
    sync_.reset(new Sync(MySyncPolicy(50), upper_velodyne_sub, lower_velodyne_sub, stitched_image_sub));
    sync_->registerCallback(boost::bind(&RegisterRGBD::SyncVeloRGB, this, _1, _2, _3));
}

void RegisterRGBD::SyncVeloRGB(const sensor_msgs::PointCloud2::ConstPtr& upper_velodyne_msg,
                                const sensor_msgs::PointCloud2::ConstPtr& lower_velodyne_msg,
                                 const sensor_msgs::Image::ConstPtr& stitched_image_msg)
{
    tf::TransformBroadcaster upper_vel, lower_vel;
    //upper_vel.sendTransform(tf::StampedTransform(upper_vel2occam, ros::Time::now(), "occam", "upper_vel"));
    //lower_vel.sendTransform(tf::StampedTransform(lower_vel2occam, ros::Time::now(), "occam", "lower_vel"));
    std::cout<<"synching\n";
    cv_ptr = cv_bridge::toCvCopy(stitched_image_msg);
    cv::Mat image = cv_ptr->image;
    //image = image*0;
    H_img = stitched_image_msg->height;
    W_img = stitched_image_msg->width;
    sensor_msgs::PointCloud2ConstIterator<float> upper_it(*upper_velodyne_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> lower_it(*lower_velodyne_msg, "x");
    for ( ; upper_it != upper_it.end(), lower_it != lower_it.end(); ++upper_it , ++lower_it ) 
    {
        // TODO: do something with the values of x, y, z
        //std::cout <<"Lower"<< lower_it[0] << ", " << lower_it[1] << ", " << lower_it[2] << '\n';
        //std::cout <<"Upper"<< upper_it[0] << ", " << upper_it[1] << ", " << upper_it[2] << '\n';
        tf::Point l_point_lower(lower_it[0],lower_it[1],lower_it[2]) , u_point_upper(upper_it[0], upper_it[1], upper_it[2]); 
        //tf::Point l_point_occam = lower_vel2occam*l_point_lower , u_point_occam = upper_vel2occam*u_point_upper;//transform lidar points from their own frames of refernce to cylindrical image frame 
        tf::Point l_point_occam = lower_vel2occam*l_point_lower , u_point_occam = upper_vel2occam*u_point_upper;//transform lidar points from their own frames of refernce to cylindrical image frame 
        std::vector<int> upper_vel_pixels = GetPixelCordinates(u_point_occam); 
        std::vector<int> lower_vel_pixels = GetPixelCordinates(l_point_occam);
        //std::cout<<"Pixels "<<upper_vel_pixels[0]<<", "<<upper_vel_pixels[1]<<" and "<<lower_vel_pixels[0]<<", "<<lower_vel_pixels[1]<<"\n";

        //std::cout<<image.at<float>(i,j)<<"\n";
        //if()
        if(upper_vel_pixels[0] <= W_img && upper_vel_pixels[1] <= H_img && upper_vel_pixels[0] >=0  && upper_vel_pixels[1] >= 0 )
        {
            if(abs(u_point_occam[2]) < 1.0)
            {
                Vec3b color; color[0] = color[1] = color[2] = abs(u_point_occam[2]) ;
            image.at<Vec3b>(upper_vel_pixels[1],upper_vel_pixels[0]) = color;}
            //image.at<float>(upper_vel_pixels[1],upper_vel_pixels[0]) = u_point_occam[2];
        }
            
        
        if(lower_vel_pixels[0] <= W_img && lower_vel_pixels[1] <= H_img && lower_vel_pixels[0] >= 0 && lower_vel_pixels[1] >= 0)
        {
            
            if(abs(l_point_occam[2]) < 1.0)
            {
            std::cout<<l_point_occam[2]<<"\n";
            Vec3b color; color[0] = color[1] = color[2] = abs(l_point_occam[2]) ;
            image.at<Vec3b>(lower_vel_pixels[1],lower_vel_pixels[0]) = color;}
            //image.at<float>(lower_vel_pixels[1],lower_vel_pixels[0]) = l_point_occam[2];
        }
            
    
    }
    cv_ptr->image = image;
    
    if(cv_ptr)
    {
        depth_encoded_pub.publish(cv_ptr->toImageMsg());
    }


}

std::vector<int> RegisterRGBD::GetPixelCordinates(const tf::Point &point)
{
    std::vector<int> pixels;
    pixels.resize(2);
    pixels[0] = W_img * (atan2(point[0], point[2]) + M_PI) / (2*M_PI); 
    pixels[1] = (f_y * point[1]*cos(atan2(point[0], point[2])) / point[2]) + y_0;
    return pixels;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "main_node");
    ros::NodeHandle node;
    RegisterRGBD register_rgbd(node);
    ros::Rate loop_rate(10);
    while (ros::ok())
	{
        //std::cout<<"while loop\n";
		ros::spinOnce();
		loop_rate.sleep();
	}
    return 0; 
}