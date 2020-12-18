## Description
ROS package used to project 3D lidar point clouds onto a panaromic RGB image in the Stanford JRDB dataset. We first estimate the camera parameters for the panaromic image which is obtained by stitching RGB images from 5 stereo camera setup to cover 360&deg field of view. 
Based on the camera parameters of the pinhole camera model, we project the Velodyne PointCloud onto the image plane as described [here](http://download.cs.stanford.edu/downloads/jrdb/Sensor_setup_JRDB.pdf)
