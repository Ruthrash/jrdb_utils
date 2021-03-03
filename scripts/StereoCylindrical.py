from calibration import OmniCalibration as ocalib 
from ip_basic.ip_basic import depth_map_utils
from ip_basic.ip_basic import vis_utils
import numpy as np 
import open3d as o3d
import rospy 
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
import torch
import ros_numpy
from cv_bridge import CvBridge, CvBridgeError
import cv2
import struct

##Pointcloud stuff 
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2 

fill_type = 'fast'
extrapolate = True
blur_type = 'gaussian'

class CylindricalCalibration:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('talker', anonymous = True)
        self.calib_obj = ocalib('calibration/')
        self.lower_vel_sub = message_filters.Subscriber('/lower_velodyne/velodyne_points',PointCloud2)
        self.upper_vel_sub = message_filters.Subscriber('/upper_velodyne/velodyne_points',PointCloud2)
        #self.upper_stitched_sub = message_filters.Subscriber('/ros_indigosdk_node/stitched_image0/image_raw',Image)
        self.upper_stitched_sub = message_filters.Subscriber('/top/image_stitched',Image)
        self.lower_stitched_sub = message_filters.Subscriber('/bottom/image_stitched',Image)
        self.projected_img_pub = rospy.Publisher('/top/projected_img', Image, queue_size=10)
        self.projected_img_bottom_pub = rospy.Publisher('/bottom/projected_img', Image, queue_size=10)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.upper_vel_sub,self.lower_vel_sub, self.upper_stitched_sub, self.lower_stitched_sub], 50, 0.5, allow_headerless=True)
        self.ts.registerCallback(self.Callback)
        self.projected_msg = Image()
        self.pcl_pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)
        self.pcl_no_holes_pub = rospy.Publisher('/no_holes_pointcloud', PointCloud2, queue_size=10)
        self.u = []
        self.v = []
        for i in range(480):
            for j in range(3760):
                self.u.append(j)
                self.v.append(i)
        #print(self.u, self.v)

        rate = rospy.Rate(20) # 20hz
        while not rospy.is_shutdown():
            rate.sleep()#pub.publish(time_msg)
		    

    def Callback(self, upper_pcl_msg, lower_pcl_msg, upper_stitched_msg, lower_stitched_msg):
        cv2_img = self.bridge.imgmsg_to_cv2(upper_stitched_msg,"bgr8")
        orig_img = self.bridge.imgmsg_to_cv2(upper_stitched_msg,"bgr8")
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
        #cl_pts = self.calib_obj.project_image_to_rect(pts)
        cloud_points = self.calib_obj.project_image_to_rect(pts)##############
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'occam'
        #create pcl from points
        scaled_polygon_pcl = pcl2.create_cloud_xyz32(header, cloud_points)
        self.pcl_pub.publish(scaled_polygon_pcl)

        
        #cloud_points = pc[0:3]
        pts = pts[ (3759>pts[:,0]) & ( 479> pts[:,1]) &(pts[:,1] > 0) & (pts[:,0] > 0)   ]
        #pts_rgb = cv2_img[]
        cv2_img = cv2_img*0.0
        cv2_img[[pts[:,1].astype(int),pts[:,0].astype(int)]] = np.column_stack((pts[:,2],pts[:,2], pts[:,2]))
        #cv2.imwrite("/home/ruthz/top.jpg", cv2_img)
        cv2_img = np.float32(cv2_img)
        #############333print(np.count_nonzero(cv2_img[:,:,0]))
        final_depths = depth_map_utils.fill_in_fast(cv2_img[:,:,0], extrapolate=extrapolate, blur_type=blur_type)
 
        dense_points = np.column_stack((self.u, self.v, final_depths.flatten()))
        cloud_points = self.calib_obj.project_image_to_rect(dense_points)##############
        r = orig_img[self.v, self.u,2].astype(int)
        g = orig_img[self.v, self.u,1].astype(int)
        b = orig_img[self.v, self.u,0].astype(int)
        a = 255*np.ones(b.shape).astype(int)
        rgb_list = []
        for i in range(len(r)):
            rgb = struct.unpack('I', struct.pack('BBBB', b[i], g[i], r[i], a[i]))[0]
            rgb_list.append(rgb)
        rgb = np.array(rgb_list)
        print(cloud_points.shape, r.shape, g.shape, b.shape, rgb.shape)
        cloud_points = np.column_stack((cloud_points, rgb))
        cloud_points = cloud_points.astype(object)
        cloud_points[:,3] = cloud_points[:,3].astype(int)
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1),
          ]
        #print(cloud_points)
        #print(fields)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'occam'
        #create pcl from points
        scaled_polygon_pcl = pcl2.create_cloud(header,fields, cloud_points)
        self.pcl_no_holes_pub.publish(scaled_polygon_pcl)

        ###################3print(np.count_nonzero(final_depths), final_depths.shape)
        #cv2.imwrite("/home/ruthz/before.png",cv2_img[:,:,0]*255)
        #cv2.imwrite("/home/ruthz/after.png",final_depths*255)
        cv2_img = cv2_img.astype(np.uint8)
        new_img_msg = self.bridge.cv2_to_imgmsg(cv2_img,encoding="bgr8")
        self.projected_img_pub.publish(new_img_msg)
        #cloud_points = self.calib_obj.project_image_to_rect(pts)



        """
        cv2_img = self.bridge.imgmsg_to_cv2(lower_stitched_msg,"bgr8")
        pc_upper = ros_numpy.numpify(upper_pcl_msg).astype({'names':['x','y','z','intensity','ring'], 'formats':['f4','f4','f4','f4','f4'], 'offsets':[0,4,8,16,20], 'itemsize':32})
        pc_lower = ros_numpy.numpify(lower_pcl_msg).astype({'names':['x','y','z','intensity','ring'], 'formats':['f4','f4','f4','f4','f4'], 'offsets':[0,4,8,16,20], 'itemsize':32})
        pc_upper = torch.from_numpy(pc_upper.view(np.float32).reshape(pc_upper.shape + (-1,)))[:, [0,1,2,4]]
        pc_lower = torch.from_numpy(pc_lower.view(np.float32).reshape(pc_lower.shape + (-1,)))[:, [0,1,2,4]]
        pc_upper = self.calib_obj.move_lidar_to_lower_camera_frame(pc_upper, upper=True)
        pc_lower = self.calib_obj.move_lidar_to_lower_camera_frame(pc_lower, upper=False)
        pc = torch.cat([pc_upper, pc_lower], dim = 0)
        pc[:, 3] = 1
        pts = self.calib_obj.project_ref_to_image_torch(pc)
        pts = pts.cpu().detach().numpy()
        pts = pts.astype(int)
        pts = pts[ (3759>pts[:,0]) & ( 479> pts[:,1]) &(pts[:,1] > 0) & (pts[:,0] > 0)   ]
        cv2_img = cv2_img*0.0
        cv2_img[[pts[:,1].astype(int),pts[:,0].astype(int)]] = np.column_stack((pts[:,2],pts[:,2], pts[:,2]))
        cv2.imwrite("/home/ruthz/bottom.jpg", cv2_img)
        new_img_msg = self.bridge.cv2_to_imgmsg(cv2_img,encoding="passthrough")
        self.projected_img_bottom_pub.publish(new_img_msg)
        """

##https://answers.ros.org/question/207071/how-to-fill-up-a-pointcloud-message-with-data-in-python/


def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    assert(points.shape == colors.shape)

    buf = []

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True;
    msg.data = xyzrgb.tostring()

    return msg 







        
if __name__ == '__main__':
    obj = CylindricalCalibration()