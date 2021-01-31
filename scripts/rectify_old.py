import cv2
import numpy as np

left_img = cv2.imread("/home/ruthz/Desktop/stereo/left/0.jpeg")
right_img = cv2.imread("/home/ruthz/Desktop/stereo/right/0.jpeg")
#left_img = cv2.imread("/home/ruthz/stereoDepth/data/leftFixedStereo/left1.png")
#right_img = cv2.imread("/home/ruthz/stereoDepth/data/rightFixedStereo/right1.png")


#left_img = cv2.imread("/home/ruthz/Downloads/aer_1515_assignment/training/left/000002.png")
#right_img = cv2.imread("/home/ruthz/Downloads/aer_1515_assignment/training/right/000002.png")




# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(left_img, None)
kp2, des2 = sift.detectAndCompute(right_img, None)




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
    if m.distance < 0.7*n.distance:
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

h1, w1 = left_img.shape[0:2]
h2, w2 = right_img.shape[0:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1))


img1_rectified = cv2.warpPerspective(left_img, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(right_img, H2, (w2, h2))

#img1_rectified = cv2.rotate(img1_rectified, cv2.cv2.ROTATE_90_CLOCKWISE)
#img2_rectified = cv2.rotate(img2_rectified, cv2.cv2.ROTATE_90_CLOCKWISE)


cv2.imwrite("/home/ruthz/rectified_1.png", img1_rectified)
cv2.imwrite("/home/ruthz/rectified_2.png", img2_rectified)

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



import bagpy
from bagpy import bagreader
import pandas as pd

b = bagreader('/home/ruthz/Desktop/jrdb_eval/bags/huang-lane-2019-02-12_0.bag')

# replace the topic name as per your need
LASER_MSG = b.message_by_topic('/upper_velodyne/velodyne_points')
"""
