#!/usr/bin/env python3
	##########################################################################################
	# This script is used to synch lower image topics to the timestamps for			 # 
	# which we have the upper camera images and lidar Pointclouds and store it in a directory#
	##########################################################################################

import json 
import glob 
import sys
from cv_bridge import CvBridge, CvBridgeError
import rospy 
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import os
import cv2



def callback(sensor1_img, sensor3_img, sensor5_img, sensor7_img, sensor9_img, stmp_msg):
	print("Received an image!")
	print(seq_name)
	global time_step_idx, time_stamp, pub
	# Convert your ROS Image message to OpenCV2
	cv2_img_1 = bridge.imgmsg_to_cv2(sensor1_img,"bgr8")
	cv2_img_3 = bridge.imgmsg_to_cv2(sensor3_img,"bgr8")
	cv2_img_5 = bridge.imgmsg_to_cv2(sensor5_img,"bgr8")
	cv2_img_7 = bridge.imgmsg_to_cv2(sensor7_img,"bgr8")
	cv2_img_9 = bridge.imgmsg_to_cv2(sensor9_img,"bgr8")
	file_name = file_names[time_step_idx]
	file_name = file_name.split("/")
	file_name = file_name[len(file_name) - 1]
	file_name = file_name.split(".")
	file_name = file_name[0]

	# Save your OpenCV2 image as a jpeg 
	cv2.imwrite(save_base_dir+"/image_1/"+seq_name+"/"+file_name+".jpg", cv2_img_1)
	cv2.imwrite(save_base_dir+"/image_3/"+seq_name+"/"+file_name+".jpg", cv2_img_3)
	cv2.imwrite(save_base_dir+"/image_5/"+seq_name+"/"+file_name+".jpg", cv2_img_5)
	cv2.imwrite(save_base_dir+"/image_7/"+seq_name+"/"+file_name+".jpg", cv2_img_7)
	cv2.imwrite(save_base_dir+"/image_9/"+seq_name+"/"+file_name+".jpg", cv2_img_9)
	time_step_idx = time_step_idx + 1
	t = rospy.Time.from_sec(time_stamp[time_step_idx])##current time stamp that we want to synch to
	time_msg = Clock()
	time_msg.clock = t
	hello_str = "publishing index %d" % time_step_idx
	rospy.loginfo(hello_str)
	pub.publish(time_msg)
	

def stamp_publisher():##publishes timestamps of upper row camera images and pointclouds in the dataset



	output_folder = save_base_dir+"/image_1/"+seq_name	
	os.mkdir(output_folder)
	output_folder = save_base_dir+"/image_3/"+seq_name	
	os.mkdir(output_folder)
	output_folder = save_base_dir+"/image_5/"+seq_name	
	os.mkdir(output_folder)
	output_folder = save_base_dir+"/image_7/"+seq_name	
	os.mkdir(output_folder)
	output_folder = save_base_dir+"/image_9/"+seq_name	
	os.mkdir(output_folder)
	global time_stamp
	time_stamp = []
	time_stamp_file = seq_folders[folder_idx]+'frames_img.json'
	f = open(time_stamp_file)
	full_data = json.load(f)
	full_data = full_data['data']
	for idx in range(len(full_data)):
		time_stamp.append(full_data[idx]['timestamp'])

	rospy.init_node('talker', anonymous=True)
	global pub
	pub = rospy.Publisher('timestamp_publisher', Clock, queue_size=10)
	sensor1_img_sub = message_filters.Subscriber('/ros_indigosdk_node/image1/image_raw',Image)
	sensor3_img_sub = message_filters.Subscriber('/ros_indigosdk_node/image3/image_raw',Image)
	sensor5_img_sub = message_filters.Subscriber('/ros_indigosdk_node/image5/image_raw',Image)
	sensor7_img_sub = message_filters.Subscriber('/ros_indigosdk_node/image7/image_raw',Image)
	sensor9_img_sub = message_filters.Subscriber('/ros_indigosdk_node/image9/image_raw',Image)
	time_stamp_sub = message_filters.Subscriber('timestamp_publisher',Clock)

	ts = message_filters.ApproximateTimeSynchronizer([sensor1_img_sub, sensor3_img_sub, sensor5_img_sub, sensor7_img_sub, sensor9_img_sub,time_stamp_sub], 50, 0.5, allow_headerless=True)

	ts.registerCallback(callback)
	rate = rospy.Rate(20) # 20hz
	while not rospy.is_shutdown():
		t = rospy.Time.from_sec(time_stamp[time_step_idx])##current time stamp that we want to synch to
		time_msg = Clock()
		time_msg.clock = t
		hello_str = "publishing index %d" % time_step_idx
		rospy.loginfo(hello_str)
		pub.publish(time_msg)
		rate.sleep()

if __name__ == '__main__':

	base_dir = "/home/ruthz/Desktop/timestamps"
	save_base_dir = '/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images'
	seq_folders = glob.glob(base_dir+'/*/')
	seq_folders.sort()

	bridge = CvBridge()

	global time_step_idx
	time_step_idx = 0

	folder_idx =  int(sys.argv[1])
	print(folder_idx)
	seq_name = seq_folders[folder_idx]
	seq_name = seq_name.split("/")
	seq_name = seq_name[len(seq_name)-2]
	print(seq_name)

	file_names = glob.glob(save_base_dir+"/image_0/"+seq_name+"/*.jpg")
	file_names.sort()
	try:
		stamp_publisher()
	except rospy.ROSInterruptException:
		pass


	


"""

python3 /home/ruthz/Desktop/timestamp.py $idx & 
    rosbag play $filename --clock --rate=0.25

#!/bin/bash 
declare -i idx=0
for filename in /media/ruthz/data/cvgl/group/jrdb/data/train_dataset/rosbags/*.bag; do

echo $filename
echo $idx
roslaunch jrdb_utils sync_and_save_bottom_images.launch  file_name:=$filename number:=$idx        

idx=$((idx+1))	
done

"""	
		
