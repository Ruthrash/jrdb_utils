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

imgL = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_top_stitched/bytes-cafe-2019-02-07_0/000000.jpg",cv2.IMREAD_GRAYSCALE)  # left image
imgR = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_bottom_stitched/bytes-cafe-2019-02-07_0/000000.jpg",cv2.IMREAD_GRAYSCALE)  # right image


imgL = cv2.rotate(imgL, cv2.cv2.ROTATE_90_CLOCKWISE)
imgR = cv2.rotate(imgR, cv2.cv2.ROTATE_90_CLOCKWISE)
imgL = imgL[0:1500,:]
imgR = imgR[0:1500,:]
imgL = imgL[1500:,:]
imgR = imgR[1500:,:]

h, w = imgL.shape[:2]

########################Feature Detection ####################################

# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

# Initiate SIFT detector
sift = cv2.SIFT_create(nfeatures=2500)
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


# Using StereoSGBM
# Set disparity parameters. Note: disparity range is tuned according to
#  specific parameters obtained through trial and error.
win_size = 15
min_disp = 0
max_disp = 64
num_disp = max_disp - min_disp  # Needs to be divisible by 16
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
mode=0,
numDisparities=num_disp,
blockSize=11,
speckleWindowSize=400,
speckleRange=120,
uniquenessRatio = 12, 
preFilterCap =31,
disp12MaxDiff=2,
P1=8 * 3 * win_size ** 2,
P2=32 * 3 * win_size ** 2,)
disparity_SGBM = stereo.compute(imgL, imgR)
#plt.imshow(cv2.rotate(disparity_SGBM, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE), "gray")
plt.imshow(disparity_SGBM, "gray")
plt.colorbar()
plt.show()

plt.subplot(121), plt.imshow(cv2.rotate(imgL, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)) 
plt.subplot(122), plt.imshow(cv2.rotate(disparity_SGBM, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)) 
plt.show() 

##################See Epilines##############

# Find epilines corresponding to points 
# in right image (second image) and 
# drawing its lines on left image 

#imgL = imgL_rectified
#imgR = imgR_rectified
sift = cv2.SIFT_create(nfeatures=2000)
kp1, des1 = sift.detectAndCompute(imgL_rectified, None)
kp2, des2 = sift.detectAndCompute(imgR_rectified, None)

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



linesLeft = cv2.computeCorrespondEpilines(pts2.reshape(-1, 
                                                           1, 
                                                           2), 
                                          2, fundamental_matrix) 
linesLeft = linesLeft.reshape(-1, 3) 
img5, img6 = drawlines(imgL_rectified, imgR_rectified,  
                       linesLeft, pts1, 
                       pts2) 
   
# Find epilines corresponding to  
# points in left image (first image) and 
# drawing its lines on right image 
linesRight = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2),  
                                           1, fundamental_matrix) 
linesRight = linesRight.reshape(-1, 3) 
  
img3, img4 = drawlines(imgR_rectified, imgL_rectified,  
                       linesRight, pts2, 
                       pts1) 
   
plt.subplot(121), plt.imshow(img5) 
plt.subplot(122), plt.imshow(img3) 
plt.show() 