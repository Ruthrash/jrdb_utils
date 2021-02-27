import cv2
import numpy as np
import matplotlib.pyplot as plt


def drawlines(img1, img2, lines, pts1, pts2): 
    
    r, c = img1.shape 
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) 
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) 
      
    for r, pt1, pt2 in zip(lines, pts1, pts2): 
          
        color = tuple(np.random.randint(0, 255, 
                                        3).tolist()) 
          
        x0, y0 = map(int, [0, -r[2] / r[1] ]) 
        x1, y1 = map(int,  
                     [c, -(r[2] + r[0] * c) / r[1] ]) 
          
        img1 = cv2.line(img1,  
                        (x0, y0), (x1, y1), color, 1) 
        img1 = cv2.circle(img1, 
                          tuple(pt1), 5, color, -1) 
        img2 = cv2.circle(img2,  
                          tuple(pt2), 5, color, -1) 
    return img1, img2 




#imgLorig = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_stitched/bytes-cafe-2019-02-07_0/000000.jpg",cv2.IMREAD_GRAYSCALE)  # left image
#imgRorig = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_bottom_stitched/bytes-cafe-2019-02-07_0/000000.jpg",cv2.IMREAD_GRAYSCALE)  # right image

#imgL = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_0/bytes-cafe-2019-02-07_0/000000.jpg",cv2.IMREAD_GRAYSCALE)  # left image
#imgR = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_1/bytes-cafe-2019-02-07_0/000000.jpg",cv2.IMREAD_GRAYSCALE)  # right image

#finalimg = finalimg*0

#imgL = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/stereo/left/75.jpg",cv2.IMREAD_GRAYSCALE)  # left image
#imgR = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/stereo/right/75.jpg",cv2.IMREAD_GRAYSCALE)  # right image
imgL = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_top_stitched/bytes-cafe-2019-02-07_0/000000.jpg",cv2.IMREAD_GRAYSCALE)  # left image
imgR = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_bottom_stitched/bytes-cafe-2019-02-07_0/000000.jpg",cv2.IMREAD_GRAYSCALE)  # right image

#imgL = cv2.flip(imgL,0)
#imgR = cv2.flip(imgR,0)

#imgL = cv2.rotate(imgL, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
#imgR = cv2.rotate(imgR, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

h, w = imgL.shape[:2]
#left_img = cv2.imread("/home/ruthz/stereoDepth/data/leftFixedStereo/left1.png")
#right_img = cv2.imread("/home/ruthz/stereoDepth/data/rightFixedStereo/right1.png")


###############Undistort images and show them#########################
"""
K_left = np.array([[476.71, 0., 350.738],
              [0., 479.505, 209.532],
              [0., 0., 1.]])
d_left = np.array([-0.336591,  0.159742, 0.00012697, -7.22557e-05, -0.0461953]) 
P_left = np.array([[0.9999939799308777, 0.0006545386859215796, 0.003402929287403822, -0.010424209758639336],
[-0.0006545192445628345, 0.9999997615814209, -6.819631835242035e-06, -3.709742307662964],
[-0.0034029330126941204, 4.5923079596832395e-06, 0.9999942183494568, -56.91765594482422]])


K_right = np.array([[478.406, 0., 353.499],
              [0., 481.322,190.225],
              [0., 0., 1.]])
d_right = np.array([-0.338422, 0.163703, -0.000376267, 7.73351e-06, -0.0479871])   
P_right  = np.array([[0.999994695186615, 0.002822051290422678, 0.0016329087084159255, -0.9035884141921997],
 [-0.0028234468773007393, 0.9999956488609314, 0.0008529311744496226, -126.85087585449219],
[-0.0016304946038872004, -0.0008575370884500444, 0.9999983310699463, -56.62558364868164]])

newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K_left, d_left, (w,h), 0)
mapx, mapy = cv2.initUndistortRectifyMap(K_left, d_left, None, newcameramatrix, (w, h), 5)
newimgL = cv2.remap(imgL, mapx, mapy, cv2.INTER_LINEAR)

newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K_right, d_right, (w,h), 0)
mapx, mapy = cv2.initUndistortRectifyMap(K_right, d_right, None, newcameramatrix, (w, h), 5)
newimgR = cv2.remap(imgR, mapx, mapy, cv2.INTER_LINEAR)


fig, (oldimg_ax, newimg_ax) = plt.subplots(1, 2)
oldimg_ax.imshow(imgL)
oldimg_ax.set_title('Original left image')
newimg_ax.imshow(newimgL)
newimg_ax.set_title('Undistorted left image')
plt.show()

fig, (oldimg_ax, newimg_ax) = plt.subplots(1, 2)
oldimg_ax.imshow(imgR)
oldimg_ax.set_title('Original rigt image')
newimg_ax.imshow(newimgR)
newimg_ax.set_title('Undistorted right image')
plt.show()
"""

#imgR = newimgR 
#imgL = newimgL
#imgL = cv2.rotate(imgL, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
#imgR = cv2.rotate(imgR, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

########################Feature Detection ####################################

# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

# Initiate SIFT detector
sift = cv2.SIFT_create(nfeatures=1000)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgL, None)
kp2, des2 = sift.detectAndCompute(imgR, None)
kp_imgL = cv2.drawKeypoints(imgL,kp1,None, flags=0)
kp_imgR = cv2.drawKeypoints(imgR,kp2,None, flags=0)

fig, (leftimg_ax, rightimg_ax) = plt.subplots(1, 2)
leftimg_ax.imshow(kp_imgL)
leftimg_ax.set_title('keypoints in left image')
rightimg_ax.imshow(kp_imgR)
rightimg_ax.set_title('keypoints in right image')
plt.show()

###################################Feature Matching#####################################################
# Match keypoints in both images
# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
#matches = flann.knnMatch(des1, des2, k=2)


#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher(crossCheck=False)

#Match decriptors in the stereo pair
print(len(des1), len(des2))
matches = bf.knnMatch(des1,des2, k=2) 
# Keep good matches: calculate distinctive image features
# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.5*n.distance:
        # Keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Calculate the fundamental matrix for the cameras
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# We select only inlier points
pts1 = pts1[inliers.ravel() == 1]
pts2 = pts2[inliers.ravel() == 1]

print(len(pts1), len(pts2))
matches_img = cv2.drawMatchesKnn(imgL,kp1,imgR,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matches_img)
plt.show()

filtered_matches_img = cv2.drawMatches(imgL,kp1,imgR,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(filtered_matches_img)
plt.show()

##########Applying horizontal matches constraint for each match



h1, w1 = imgL.shape[0:2]
h2, w2 = imgR.shape[0:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1))


imgL_rectified = cv2.warpPerspective(imgL, H1, (w1, h1))
imgR_rectified = cv2.warpPerspective(imgR, H2, (w2, h2))


#imgL_rectified = imgL
#imgR_rectified = imgR
#img1_rectified = cv2.rotate(img1_rectified, cv2.cv2.ROTATE_90_CLOCKWISE)
#img2_rectified = cv2.rotate(img2_rectified, cv2.cv2.ROTATE_90_CLOCKWISE)

fig, (rect_leftimg_ax, rect_rightimg_ax) = plt.subplots(1, 2)
rect_leftimg_ax.imshow(imgL_rectified)
rect_leftimg_ax.set_title('Stereo rectified left image')
rect_rightimg_ax.imshow(imgR_rectified)
rect_rightimg_ax.set_title('Stereo rectified right image')
plt.show()

fig, (unrect_leftimg_ax, rect_leftimg1_ax) = plt.subplots(1, 2)
unrect_leftimg_ax.imshow(imgL)
unrect_leftimg_ax.set_title('Unrectified left image')
rect_leftimg1_ax.imshow(imgL_rectified)
rect_leftimg1_ax.set_title('Stereo rectified left image')
plt.show()

fig, (unrect_rightimg_ax, rect_rightimg1_ax) = plt.subplots(1, 2)
unrect_rightimg_ax.imshow(imgR)
unrect_rightimg_ax.set_title('Unrectified right image')
rect_rightimg1_ax.imshow(imgR_rectified)
rect_rightimg1_ax.set_title('Stereo rectified right image')
plt.show()


############## Calculate Disparity (Depth Map) ##############

# Using StereoBM
prefiltertype = cv2.StereoBM_PREFILTER_XSOBEL
prefiltersize = 9
prefiltercap = 31
sad_window_size = 15
min_disp = 0
num_disp = 64
text_thresh = 10
uniqueness_ratio = 12
speckle_range = 120
speckle_window_size = 400

#stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
stereo = cv2.StereoBM_create(preFilterType = prefiltertype, 
preFilterSize = prefiltersize, 
preFilterCap = prefiltercap,
SADWindowSize = sad_window_size,
minDisparity = min_disp, 
numDisparities = num_disp,
textureThreshold = text_thresh,
uniquenessRatio = uniqueness_ratio,
speckleRange =speckle_range, 
speckleWindowSize = speckle_window_size)

disparity_BM = stereo.compute(imgL_rectified, imgR_rectified)
plt.imshow(disparity_BM, "gray")
plt.colorbar()
plt.show()

# Using StereoSGBM
# Set disparity parameters. Note: disparity range is tuned according to
#  specific parameters obtained through trial and error.
win_size = 15
min_disp = 0
max_disp = 64
num_disp = max_disp - min_disp  # Needs to be divisible by 16
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
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
disparity_SGBM = stereo.compute(imgL_rectified, imgR_rectified)
plt.imshow(disparity_SGBM, "gray")
plt.colorbar()
plt.show()

disparity_SGBM = stereo.compute(imgL_rectified, imgR_rectified).astype(np.float32) / 16.0

plt.imshow(disparity_SGBM, "gray")
plt.colorbar()
plt.show()



new_img = disparity_SGBM*0

fx = 476.71
fy = 479.505
cx = 350.738
cy = 209.532
B = 120 
for u in range(w):
    for v in range(h):
        Z = fx*B/disparity_SGBM[v,u]
        new_img[v,u] = Z


plt.imshow(new_img, "gray")
plt.colorbar()
plt.show()

plt.imshow(new_img, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()


disparity_SGBM = cv2.rotate(disparity_SGBM, cv2.cv2.ROTATE_90_CLOCKWISE)

plt.imshow(disparity_SGBM, "gray")
plt.colorbar()
plt.show()

newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K_left, d_left*-1, (w,h), 0)
undistorted_frame = cv2.undistort(disparity_SGBM, K_left, d_left*-1, None, newcameramatrix)

#imgL = cv2.rotate(imgL, cv2.cv2.ROTATE_90_CLOCKWISE)
disparity_SGBM = (disparity_SGBM - min_disp)/num_disp
inv_disp = disparity_SGBM 
inv_disp = 1- inv_disp
ret, inv_disp = cv2.threshold(invdisp,.98,255,cv2.THRESH_TOZERO_INV)
plt.imshow(inv_disp)
plt.show()

plt.subplot(121), plt.imshow(cv2.flip(undistorted_frame,1)) 
plt.subplot(122), plt.imshow(imgLorig) 
plt.show()




        #Z = abs(Z)
        #X = (u-cx)*Z/fx
        #Y = (v-cy)*Z/fy
        #p = np.array([X,Y,Z,1])
        #new_point = np.matmul(P_left, p)
        #print(new_point)


"""
X, Y = np.meshgrid(range(w1), range(h1)
pnts_distorted = np.concatenate(X, Y).reshape(w1*h1, 2)
pnts_rectified = cv2.undistortPoints(pnts_distorted, P_left, d_left)
mapx = pnts_rectified[:,:,0]
mapy = pnts_rectified[:,:,1]

img_distored = cv2.remap(disparity_SGBM, mapx, mapy, cv2.INTER_LINEAR)
"""

##################See Epilines##############

# Find epilines corresponding to points 
# in right image (second image) and 
# drawing its lines on left image 

#imgL = imgL_rectified
#imgR = imgR_rectified
linesLeft = cv2.computeCorrespondEpilines(pts2.reshape(-1, 
                                                           1, 
                                                           2), 
                                          2, fundamental_matrix) 
linesLeft = linesLeft.reshape(-1, 3) 
img5, img6 = drawlines(imgL, imgR,  
                       linesLeft, pts1, 
                       pts2) 
   
# Find epilines corresponding to  
# points in left image (first image) and 
# drawing its lines on right image 
linesRight = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2),  
                                           1, fundamental_matrix) 
linesRight = linesRight.reshape(-1, 3) 
  
img3, img4 = drawlines(imgR, imgL,  
                       linesRight, pts2, 
                       pts1) 
   
plt.subplot(121), plt.imshow(img5) 
plt.subplot(122), plt.imshow(img3) 
plt.show() 



#cv2.imwrite("/home/ruthz/rectified_1.png", img1_rectified)
#cv2.imwrite("/home/ruthz/rectified_2.png", img2_rectified)

"""
orb = cv2.ORB_create(nfeatures=200)

kp1, des1 = orb.detectAndCompute(left_img,None)
print("left image has "+str(len(des1))+" features")
kp2, des2 = orb.detectAndCompute(right_img,None)
print("right image has "+str(len(des2))+" features")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#Match decriptors in the stereo pair
matches = bf.match(des1,des2)  
matches = sorted(matches, key = lambda x:x.distance)


img2 =	cv2.drawMatches(left_img,kp1,right_img,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite( "horizontal_match_1.png", img2);

matched_left_pts = np.zeros([len(matches),2])
matched_right_pts = np.zeros([len(matches),2])
pts1 = []
pts2 = []
for j in range(len(matches)):
	#matched_left_pts[j] = cv2.KeyPoint_convert(kp1)[matches[j].queryIdx]
	#matched_right_pts[j] = cv2.KeyPoint_convert(kp2)[matches[j].trainIdx]
	pts1.append(kp1[matches[j].queryIdx].pt)
	pts2.append(kp2[matches[j].trainIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)


fundamental_matrix,inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)#2, 2, 0.9999);##fundamental matrix computed using RANSAC

# We select only inlier points
pts1 = pts1[inliers.ravel() == 1]
pts2 = pts2[inliers.ravel() == 1]

h1, w1 = left_img.shape[0:2]
h2, w2 = right_img.shape[0:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
)


img1_rectified = cv2.warpPerspective(left_img, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(right_img, H2, (w2, h2))

#img1_rectified = cv2.rotate(img1_rectified, cv2.cv2.ROTATE_90_CLOCKWISE)
#img2_rectified = cv2.rotate(img2_rectified, cv2.cv2.ROTATE_90_CLOCKWISE)


cv2.imwrite("/home/ruthz/rectified_1.png", img1_rectified)
cv2.imwrite("/home/ruthz/rectified_2.png", img2_rectified)

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask[300:500],
                   flags=cv.DrawMatchesFlags_DEFAULT)

keypoint_matches = cv.drawMatchesKnn(
    left_img, kp1, right_img, kp2, matches[300:500], None, **draw_params)
cv.imshow("Keypoint matches", keypoint_matches)





points = [[-0.0104242, -3.70974, -56.9177],
	 [0.93957, -4.05131, -52.03],
	[-0.25753, -6.54978, -47.7311],
	[2.72207, -6.82928, -45.9778],
	[-0.333857, -5.12974, -56.0573],[-0.903588, -126.851, -56.6256],
	[1.74525, -127.214, -51.7722],[-2.56535, -129.191, -47.5803],
	[ 3.39727, -129.381, -45.2409],[0.354966, -128.218, -54.0617]]


import bagpy
from bagpy import bagreader
import pandas as pd

b = bagreader('/home/ruthz/Desktop/jrdb_eval/bags/huang-lane-2019-02-12_0.bag')

# replace the topic name as per your need
LASER_MSG = b.message_by_topic('/upper_velodyne/velodyne_points')

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity_BM = stereo.compute(imgL_rectified, imgR_rectified)#, disptype=cv2.CV_32F)
norm_coeff = 255 / disparity_BM.max()
cv2.imshow("disparity", disparity_BM * norm_coeff / 255)


plt.imshow(disparity_BM, "gray")
plt.colorbar()
plt.show()

# Using StereoSGBM
# Set disparity parameters. Note: disparity range is tuned according to
#  specific parameters obtained through trial and error.
win_size = 5
min_disp = -6
max_disp = 42
num_disp = max_disp - min_disp  # Needs to be divisible by 16
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    mode=StereoSGBM::MODE_HH,
    numDisparities=num_disp,
    blockSize=11,
    uniquenessRatio=7,
    speckleWindowSize=100,
    speckleRange=1,
    disp12MaxDiff=2,
    P1=8 * 3 * win_size ** 2,
    P2=32 * 3 * win_size ** 2,
)
disparity_SGBM = stereo.compute(imgL_rectified, imgR_rectified, disptype=cv2.CV_32F)
norm_coeff = 255 / disparity_SGBM.max()
cv2.imshow("disparity", disparity_SGBM * norm_coeff / 255)
cv2.waitKey(0)
"""
