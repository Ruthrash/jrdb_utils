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
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2 
import torch
import ros_numpy 
import std_msgs.msg
def DrawKeypoint(img, kps):
    for kp in kps:
        cv2.drawMarker(img, (kp[0], kp[1]),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2, line_type=cv2.LINE_AA)
    return img

class OpenPose3DNode:
    def __init__(self, model_folder):
        self.viz_ = viz(ns='visualization', skeleton_frame='occam',id_text_size=0.2,id_text_offset=-0.05, skeleton_line_width=0.07)
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
        self.lower_vel_sub = message_filters.Subscriber('/lower_velodyne/velodyne_points',PointCloud2)
        self.upper_vel_sub = message_filters.Subscriber('/upper_velodyne/velodyne_points',PointCloud2)   
        self.pcl_pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=1)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.upper_stitched_sub, self.lower_stitched_sub, self.lower_vel_sub, self.upper_vel_sub], 50, 0.5, allow_headerless=True)
        self.ts.registerCallback(self.StichedImagesCallback)
        rate =  rospy.Rate(20)
        while not rospy.is_shutdown():
            rate.sleep()

    def StichedImagesCallback(self, image_top_stitched_msg, image_bottom_stitched_msg, lower_pcl_msg, upper_pcl_msg):
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
            theta = (imgTopKp[top_idx,:,0]/3760.0)*2*np.pi - np.pi
            for i in range(len(depth)):
                if(disparity[i]!=0):
                    depth[i] = abs(f_y*B/(disparity[i]*1000.0))
                    pose_3d[i,0] = depth[i]*np.sin(theta[i])
                    pose_3d[i,2] = depth[i]*np.cos(theta[i])
                    pose_3d[i,1] = pose_3d[i,2]*-1*(imgTopKp[top_idx,i,1] - c_y)/(f_y*np.cos(theta[i]))
                    
                else:
                    depth[i] = 0
                    pose_3d[i,0] = 0
                    pose_3d[i,1] = 0
                    pose_3d[i,2] = 0
            poses_3d.append(pose_3d)
            #print(pose_3d.shape)
        pc_upper = ros_numpy.numpify(upper_pcl_msg).astype({'names':['x','y','z','intensity','ring'], 'formats':['f4','f4','f4','f4','f4'], 'offsets':[0,4,8,16,20], 'itemsize':32})
        pc_lower = ros_numpy.numpify(lower_pcl_msg).astype({'names':['x','y','z','intensity','ring'], 'formats':['f4','f4','f4','f4','f4'], 'offsets':[0,4,8,16,20], 'itemsize':32})
        pc_upper = torch.from_numpy(pc_upper.view(np.float32).reshape(pc_upper.shape + (-1,)))[:, [0,1,2,4]]
        pc_lower = torch.from_numpy(pc_lower.view(np.float32).reshape(pc_lower.shape + (-1,)))[:, [0,1,2,4]]
        ##lidar to camera frame
        pc_upper = self.calib_obj.move_lidar_to_camera_frame(pc_upper, upper=True)
        pc_lower = self.calib_obj.move_lidar_to_camera_frame(pc_lower, upper=False)
        pc = torch.cat([pc_upper, pc_lower], dim = 0)
        pc[:, 3] = 1
        pts = self.calib_obj.project_ref_to_image_torch(pc)#################
        pts = pts.cpu().detach().numpy()
        pts = pts[ (3759>pts[:,0]) & ( 479> pts[:,1]) &(pts[:,1] > 0) & (pts[:,0] > 0)   ]

        cloud_points = self.calib_obj.project_image_to_rect(pts)##############
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'occam'
        #create pcl from points
        scaled_polygon_pcl = pcl2.create_cloud_xyz32(header, cloud_points)
        self.pcl_pub.publish(scaled_polygon_pcl)        
        self.viz_.PublishSkeleton(poses_3d) 
            

            
if __name__ == '__main__':
    obj = OpenPose3DNode("/home/ruthz/openpose/models/")