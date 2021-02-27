import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

sys.path.append('/home/ruthz/openpose/build/python')
from openpose import pyopenpose as op

from scipy.optimize import linear_sum_assignment#hungarian algorithm

from cv_bridge import CvBridge, CvBridgeError
import rospy
import message_filters
from sensor_msgs.msg import Image
from ros_openpose.msg import Frame
from calibration import OmniCalibration as ocalib 
from Visualizer3D import RealtimeVisualization as viz

def DrawKeypoint(img, kps):
    for kp in kps:
        cv2.drawMarker(img, (kp[0], kp[1]),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2, line_type=cv2.LINE_AA)
    return img

class OpenPose3DNode:
    def __init__(self, model_folder):
        self.viz_ = viz(ns='visualization', skeleton_frame='occam',id_text_size=0.2,id_text_offset=-0.05, skeleton_line_width=0.01)
        self.calib_obj = ocalib('calibration/')
        self.params = dict()
        self.params["model_folder"] = model_folder#
        self.params["face"] = False
        self.params["hand"] = False
        self.params['net_resolution'] = "-1x272"
        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()
        self.datum = op.Datum()
        self.bridge = CvBridge()
        rospy.init_node('talker', anonymous = True)
        self.upper_stitched_sub = message_filters.Subscriber('/top/image_stitched',Image)
        self.lower_stitched_sub = message_filters.Subscriber('/bottom/image_stitched',Image)    
        self.ts = message_filters.ApproximateTimeSynchronizer([self.upper_stitched_sub, self.lower_stitched_sub], 50, 0.5, allow_headerless=True)
        self.ts.registerCallback(self.StichedImagesCallback)
        rate =  rospy.Rate(20)
        while not rospy.is_shutdown():
            rate.sleep()

    def StichedImagesCallback(self, image_top_stitched_msg, image_bottom_stitched_msg):
        top_stitched_img = self.bridge.imgmsg_to_cv2(image_top_stitched_msg,"bgr8")
        bottom_stitched_img = self.bridge.imgmsg_to_cv2(image_bottom_stitched_msg,"bgr8")

        self.datum.cvInputData = top_stitched_img
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        #imgTopDet = self.datum.cvOutputData
        imgTopKp = self.datum.poseKeypoints
        #imgTopDet = cv2.rotate(imgTop,cv2.cv2.ROTATE_90_CLOCKWISE)
        self.datum.cvInputData = bottom_stitched_img
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        #imgBottomDet = self.datum.cvOutputData
        imgBottomKp = self.datum.poseKeypoints
        
        f_y = 482.924
        f_x = 480.459
        c_x = 209.383
        c_y = 355.144 
        r = 3360000
        B = 120
        #####Hungarian algorithm to match openPose detections####
        cost = np.zeros((len(imgTopKp), len(imgBottomKp)))
        for Topidx in range(len(imgTopKp)):
            for Bottomidx in range(len(imgBottomKp)):
                cost[Topidx, Bottomidx] = sum(abs(imgTopKp[Topidx,:,0] - imgBottomKp[Bottomidx,:,0]))

        ### Pose detection matching across two cameras and visualization ###
        row_idxs , col_idxs = linear_sum_assignment(cost)
        poses_3d = list()
        for top_idx, bottom_idx in zip(row_idxs, col_idxs):
            pose_3d = imgTopKp[top_idx]
            disparity = imgTopKp[top_idx,:,1] - imgBottomKp[bottom_idx,:,1]
            depth = disparity*0
            for i in range(len(depth)):
                if(disparity[i]!=0):
                    depth[i] = abs(f_y*B/(disparity[i]*1000.0))
                    pose_3d[i,0] = (pose_3d[i,0] - c_x)*depth[i]/f_x
                    pose_3d[i,1] = (pose_3d[i,1] -c_y)*depth[i]/f_y#*depth[i]
                    pose_3d[i,2] = depth[i]
                else:
                    depth[i] = 0
                    pose_3d[i,0] = 0
                    pose_3d[i,1] = 0
                    pose_3d[i,2] = 0
            poses_3d.append(pose_3d)
            #print(pose_3d.shape)
        self.viz_.PublishSkeleton(poses_3d) 
            

            
if __name__ == '__main__':
    obj = OpenPose3DNode("/home/ruthz/openpose/models/")