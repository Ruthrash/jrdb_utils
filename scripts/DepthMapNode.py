from DepthImageProc import DepthImageProc as d_image_proc
import numpy as np
import rospy
#from calibration import Calibration as calib
#calibration_path = 'calibration/cameras.yaml'
#calib_param = calib

K_0 = np.array([[476.71, 0, 350.738],
      [0, 479.505, 209.532],
      [0, 0, 1]])
d_0 = np.array([-0.336591, 0.159742, 0.00012697, -7.22557e-05, -0.0461953])

K_1 = np.array([[483.254, 0, 365.33],
      [0, 485.78, 210.953],
      [0 ,0, 1]])
d_1 = np.array([-0.335073, 0.151959, -0.000232061, 0.00032014, -0.0396825])

K_2 = np.array([[483.911, 0, 355.144],
      [0, 486.466, 223.026],
      [0, 0, 1]])
d_2 = np.array([-0.33846, 0.156256, -0.000385467, 0.000295485, -0.0401965])

K_3 = np.array([[475.807, 0, 339.53],
      [0, 478.371, 188.481],
      [0, 0, 1]])
d_3 = np.array([-0.330848, 0.14747, 8.59247e-05, 0.000262599, -0.0385311])

K_4 = np.array([[485.046, 0, 368.864],
      [0, 488.185, 208.215],
      [0, 0, 1]])
d_4 = np.array([-0.34064, 0.168338, 0.000147292, 0.000229372, -0.0516133])

K_5 = np.array([[478.406, 0, 353.499],
      [0, 481.322, 190.225],
      [0, 0, 1]])
d_5 = np.array([ -0.338422, 0.163703, -0.000376267, 7.73351e-06, -0.0479871])

K_6 = np.array([[480.459, 0, 362.503],
      [0, 482.924, 197.949],
      [0, 0, 1]])
d_6 = np.array([-0.340676, 0.165511, -0.00035978, 0.000181532, -0.0493721])

K_7 = np.array([[486.491, 0, 361.559],
      [0, 489.22, 210.547],
      [0, 0, 1]])
d_7 = np.array([-0.344379, 0.170343, -0.000137847, 0.000141047, -0.0510536])

K_8 = np.array([[476.708, 0, 354.16],
      [0, 479.424, 209.383],
      [0, 0, 1]])
d_8 = np.array([-0.331228, 0.144696, 0.000117553, 0.000566449, -0.0343506])

K_9 = np.array([[484.219, 0, 345.303],
      [0, 487.312, 192.371],
      [0, 0, 1]])
d_9 = np.array([-0.345189, 0.180808, 0.000276465, 0.000131868, -0.062103])

def NodeMain():##publishes timestamps of upper row camera images and pointclouds in the dataset
    rospy.init_node('listener', anonymous=True)
    stereo_0 = d_image_proc(K_0, d_0, K_5, d_5, 0, 1)
    stereo_2 = d_image_proc(K_1, d_1, K_6, d_6, 2, 3)
    stereo_4 = d_image_proc(K_2, d_2, K_7, d_7, 4, 5)
    stereo_6 = d_image_proc(K_3, d_3, K_8, d_8, 6, 7)
    stereo_8 = d_image_proc(K_4, d_4, K_9, d_9, 8, 9)
    
    rospy.spin()
    """
	rate = rospy.Rate(20) # 20hz
	while not rospy.is_shutdown():
		t = rospy.Time.from_sec(time_stamp[time_step_idx])##current time stamp that we want to synch to
		time_msg = Clock()
		time_msg.clock = t
		hello_str = "publishing index %d" % time_step_idx
		rospy.loginfo(hello_str)
		pub.publish(time_msg)
		rate.sleep()
    """













if __name__ == '__main__':
	try:
		NodeMain()
	except rospy.ROSInterruptException:
		pass



