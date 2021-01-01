## Description
ROS package used to project 3D lidar point clouds onto a panaromic RGB image in the Stanford JRDB dataset.
Using the camera parameters of the indvidual pinhole camera model, we project the Velodyne PointCloud onto the image plane as described [here](http://download.cs.stanford.edu/downloads/jrdb/Sensor_setup_JRDB.pdf)


## Usage 

```
rosrun jrdb_utils register_rgbd
```
To uncompress images in the ROSbags
```
rosrun image_transport republish compressed in:=/ros_indigosdk_node/stitched_image0 raw out:=/ros_indigosdk_node/stitched_image0/image_raw

```
To view projected image 

```
rosrun image_view image_view image:=/depth_moed_pub 

```
