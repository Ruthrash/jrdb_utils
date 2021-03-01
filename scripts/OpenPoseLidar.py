import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import ros_numpy
import torch


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

##Pointcloud stuff 
from ip_basic.ip_basic import depth_map_utils
from ip_basic.ip_basic import vis_utils
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2 

fill_type = 'fast'
extrapolate = True
blur_type = 'gaussian'

def DrawKeypoint(img, kps):
    for kp in kps:
        if(kp[0] !=0 and kp[1] !=0):
            cv2.drawMarker(img, (kp[0], kp[1]),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
    return img

class OpenPose3DNode:
    def __init__(self, model_folder):
        self.viz_ = viz(ns='visualization', skeleton_frame='occam',id_text_size=0.2,id_text_offset=-0.05, skeleton_line_width=0.1)
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
        self.lower_vel_sub = message_filters.Subscriber('/lower_velodyne/velodyne_points',PointCloud2)
        self.upper_vel_sub = message_filters.Subscriber('/upper_velodyne/velodyne_points',PointCloud2)
        self.upper_stitched_sub = message_filters.Subscriber('/top/image_stitched',Image)
        self.pcl_pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=1)
        self.proj_pub = rospy.Publisher('/projected_keypoints', Image, queue_size=1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.upper_stitched_sub, self.lower_vel_sub, self.upper_vel_sub  ], 50, 0.5, allow_headerless=True)
        self.ts.registerCallback(self.StichedImagesCallback)
        rate =  rospy.Rate(20)
        while not rospy.is_shutdown():
            rate.sleep()

    def StichedImagesCallback(self, image_top_stitched_msg, lower_pcl_msg, upper_pcl_msg):
        top_stitched_img = self.bridge.imgmsg_to_cv2(image_top_stitched_msg,"bgr8")
        ##forward pass through openpose 
        self.datum.cvInputData = top_stitched_img
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        saved_img = self.datum.cvOutputData
        imgTopKp = self.datum.poseKeypoints



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
        #saved_img = top_stitched_img

        top_stitched_img = top_stitched_img*0.0
        top_stitched_img[[pts[:,1].astype(int),pts[:,0].astype(int)]] = np.column_stack((pts[:,2],pts[:,2], pts[:,2]))
        #cv2.imwrite("/home/ruthz/top.jpg", top_stitched_img)
        top_stitched_img = np.float32(top_stitched_img)
        #############333print(np.count_nonzero(top_stitched_img[:,:,0]))
        final_depths = depth_map_utils.fill_in_fast(top_stitched_img[:,:,0], extrapolate=extrapolate, blur_type=blur_type)
        #kp_depths = final_depths[[imgTopKp[:,1].astype(int), imgTopKp[:,0].astype(int)    ]]
        #[pts[:,1].astype(int),pts[:,0].astype(int)]
        poses_3d = list()
        for person_kps in imgTopKp:
            kp_depths = final_depths[ [person_kps[:,1].astype(int), person_kps[:,0].astype(int)] ] 
            person_uvdepth = np.column_stack((person_kps[:,0].astype(int), person_kps[:,1].astype(int), kp_depths))
            ###set depth of (0,0) keypoint detections as zero 
            person_uvdepth [np.logical_and(person_uvdepth[:,0]==0,person_uvdepth[:,0]==0)] = np.zeros(person_uvdepth [np.logical_and(person_uvdepth[:,0]==0,person_uvdepth[:,0]==0)].shape)
            person_3d_kp = self.calib_obj.project_image_to_rect(person_uvdepth)
            kp_3d = torch.from_numpy(person_3d_kp)
           # print(kp_3d)
            pose_kp = self.calib_obj.project_ref_to_image_torch(kp_3d)
            pose_kp = pose_kp.cpu().detach().numpy()
            pose_kp[ np.isnan(pose_kp[:,1])] = np.zeros(pose_kp[ np.isnan(pose_kp[:,1])].shape)
            pose_kp = pose_kp.astype(int)
            saved_img = DrawKeypoint(saved_img, pose_kp)
            #print(person_3d_kp[0])
            poses_3d.append(person_3d_kp)
            #print(person_uvdepth[person_uvdepth[:,0:2]==[0,0]]) 
            #print(person_uvdepth[person_uvdepth[:,0] == [0,0]] )
            #print(kp_depths.shape, person_kps.shape, person_uvdepth.shape)

        new_img_msg = self.bridge.cv2_to_imgmsg(saved_img,encoding="bgr8")
        self.proj_pub.publish(new_img_msg)
        self.viz_.PublishSkeleton(poses_3d)

            
if __name__ == '__main__':
    obj = OpenPose3DNode("/home/ruthz/openpose/models/")