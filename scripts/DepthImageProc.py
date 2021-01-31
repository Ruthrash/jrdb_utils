#!/usr/bin/env python3
	##########################################################################################
	# This script is used to synch rectified pair of upper and lower images and republishes  #
    # depth images for stitching                                                             #        
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
import numpy as np


class DepthImageProc:

    def __init__(self, K_left, d_left, K_right, d_right, camera_left_id, camera_right_id):
        self.camera_left_id = camera_left_id ; self.camera_right_id = camera_right_id
        self.depth_pub = rospy.Publisher('depth_image_'+str(camera_left_id), Image, queue_size=10)
        self.left_img_sub = message_filters.Subscriber('/ros_indigosdk_node/image'+str(camera_left_id)+'/image_raw',Image)
        self.right_img_sub = message_filters.Subscriber('/ros_indigosdk_node/image'+str(camera_right_id)+'/image_raw',Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.left_img_sub, self.right_img_sub], 50, 0.5, allow_headerless=True)
        self.ts.registerCallback(self.RectifyPairCallback)
        self.K_left = K_left;  self. d_left = d_left
        self.K_right = K_right; self.d_right = d_right
        self.bridge = CvBridge()
        win_size = 15
        min_disp = 0
        max_disp = 64
        num_disp = max_disp - min_disp  # Needs to be divisible by 16
        self.stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
        #mode=StereoSGBM::MODE_HH,
        numDisparities=num_disp,
        blockSize=11,
        speckleWindowSize=400,
        speckleRange=120,
        uniquenessRatio = 12, 
        preFilterCap =31,
        disp12MaxDiff=2,
        P1=8 * 3 * win_size ** 2,
        P2=32 * 3 * win_size ** 2,)

    def RectifyPairCallback(self, left_img_msg, right_img_msg):
        print("getting images of "+str(self.camera_left_id)+" and "+str(self.camera_right_id))
        imgL =  self.bridge.imgmsg_to_cv2(left_img_msg,"bgr8")
        imgR =  self.bridge.imgmsg_to_cv2(right_img_msg,"bgr8")
        self.header = left_img_msg.header
        self.PublishDistortedDisparityImage(imgL, imgR)
        

    def PublishDistortedDisparityImage(self, imgL, imgR):
        disparity_SGBM = self.stereo.compute(imgL, imgR)
        #depth_SGBM = disparity_SGBM*0
        h, w = imgL.shape[:2]
        depth_SGBM = np.zeros((h, w, 1), np.uint16)
        fx = self.K_left[0,0]; fy = self.K_left[1,1]
        cx = self.K_left[0,2]; cy = self.K_left[1,2]; B = 120 
        for u in range(w):
            for v in range(h):
                if(disparity_SGBM[v,u] != 0):
                    Z = fx*B/disparity_SGBM[v,u]
                    depth_SGBM[v,u] = abs(Z)

        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(self.K_left, self.d_left*-1, (w,h), 0)
        distorted_depth_img = cv2.undistort(depth_SGBM, self.K_left, self.d_left*-1, None, newcameramatrix)
        ditorted_depth_img   = cv2.flip(distorted_depth_img,1)
        distorted_depth_msg = self.bridge.cv2_to_imgmsg(distorted_depth_img, encoding="mono16")
        distorted_depth_msg.header = self.header
        try:
            self.depth_pub.publish(distorted_depth_msg)
        except CvBridgeError as e:
            print(e)