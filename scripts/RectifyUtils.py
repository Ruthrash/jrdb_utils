import cv2
import numpy as np
import matplotlib.pyplot as plt
imgL = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_0/bytes-cafe-2019-02-07_0/000000.jpg",cv2.IMREAD_GRAYSCALE)  # right image
#imgL = cv2.rotate(imgL, cv2)
imgR = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_1/bytes-cafe-2019-02-07_0/000000.jpg",cv2.IMREAD_GRAYSCALE)  # right image
#imgL = cv2.rotate(imgL, cv2)
#imgL = cv2.imread("/home/ruthz/Desktop/stereo/left/0.jpeg",cv2.IMREAD_GRAYSCALE)  # left image
#imgR = cv2.imread("/home/ruthz/Desktop/stereo/right/0.jpeg",cv2.IMREAD_GRAYSCALE)  # right image
h, w = imgL.shape[:2]

#imgL = cv2.rotate(imgL, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
#imgR = cv2.rotate(imgR, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

"""

K_left = np.array([[476.71, 0., 350.738],
              [0., 479.505, 209.532],
              [0., 0., 1.]])
K_right = np.array([[483.254, 0., 365.33],
              [0., 485.78, 210.953],
              [0., 0., 1.]])
R_left = np.array([[0.999994 ,0.000654539, 0.00340293],
                    [-0.000654519 ,1, -6.81963e-06],
                    [-0.00340293, 4.59231e-06, 0.999994]])

t_left = np.array([-0.0104242, -3.70974, -56.9177])      
T_left = np.zeros((3,4))
T_left[:4,:4] = R_left ; T_left[:,3] = t_left 
P_left = np.matmul(K_left, T_left)

K_right = np.array([[478.406, 0., 353.499],
              [0., 481.322,190.225],
              [0., 0., 1.]])
d_right = np.array([-0.338422, 0.163703, -0.000376267, 7.73351e-06, -0.0479871])   
"""



d_left = np.array([-0.336591,  0.159742, 0.00012697, -7.22557e-05, -0.0461953])              
d_right = np.array([ -0.335073, 0.151959, -0.000232061, 0.00032014, -0.0396825])              
"""
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K_left, d_left, (w,h), 0)
mapx, mapy = cv2.initUndistortRectifyMap(K_left, d_left, None, newcameramatrix, (w, h), 5)
newimgL = cv2.remap(imgL, mapx, mapy, cv2.INTER_LINEAR)

newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K_right, d_right, (w,h), 0)
mapx, mapy = cv2.initUndistortRectifyMap(K_right, d_right, None, newcameramatrix, (w, h), 5)
newimgR = cv2.remap(imgR, mapx, mapy, cv2.INTER_LINEAR)


imgR = newimgR 
imgL = newimgL
#imgL = cv2.rotate(imgL, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
#imgR = cv2.rotate(imgR, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

fig, (oldimg_ax, newimg_ax) = plt.subplots(1, 2)
oldimg_ax.imshow(imgL)
oldimg_ax.set_title('Original image')
newimg_ax.imshow(newimgL)
newimg_ax.set_title('Unwarped image')
plt.show()
"""
def get_keypoints_and_descriptors(imgL, imgR):
    """Use ORB detector and FLANN matcher to get keypoints, descritpors,
    and corresponding matches that will be good for computing
    homography.
    """
    orb = cv2.ORB_create(nfeatures=1000)
    sift = cv2.SIFT_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)
    print(len(kp1))
    ############## Using FLANN matcher ##############
    # Each keypoint of the first image is matched with a number of
    # keypoints from the second image. k=2 means keep the 2 best matches
    # for each keypoint (best matches = the ones with the smallest
    # distance measurement).
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=1,
    )  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    #flann_match_pairs = flann.knnMatch(des1, des2, k=2)
    bf = cv2.BFMatcher(crossCheck=False)
    #matches = bf.knnMatch(des1,des2, k=2)
    matches = bf.match(des1,des2) 
    return kp1, des1, kp2, des2, matches


def lowes_ratio_test(matches, ratio_threshold=0.7):
    """Filter matches using the Lowe's ratio test.

    The ratio test checks if matches are ambiguous and should be
    removed by checking that the two distances are sufficiently
    different. If they are not, then the match at that keypoint is
    ignored.

    https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    """
    filtered_matches = []

    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < 0.9*n.distance:
                filtered_matches.append(m)
    
        except ValueError:	
            pass
    return filtered_matches
"""
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            filtered_matches.append(m)
    return filtered_matches
"""

def draw_matches(imgL, imgR, kp1, des1, kp2, des2, flann_match_pairs):
    """Draw the first 8 mathces between the left and right images."""
    # https://docs.opencv.org/4.2.0/d4/d5d/group__features2d__draw.html
    # https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
    img = cv2.drawMatches(
        imgL,
        kp1,
        imgR,
        kp2,
        flann_match_pairs[:8],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Matches", img)
    cv2.imwrite("ORB_FLANN_Matches.png", img)
    cv2.waitKey(0)


def compute_fundamental_matrix(matches, kp1, kp2, method=cv2.FM_RANSAC):
    """Use the set of good mathces to estimate the Fundamental Matrix.

    See  https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm
    for more info.
    """
    pts1, pts2 = [], []
    fundamental_matrix, inliers = None, None
    for m in matches[:8]:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    if pts1 and pts2:
        # You can play with the Threshold and confidence values here
        # until you get something that gives you reasonable results. I
        # used the defaults
        fundamental_matrix, inliers = cv2.findFundamentalMat(
            np.float32(pts1),
            np.float32(pts2),
            method=method,
            # ransacReprojThreshold=3,
            # confidence=0.99,
        )
    return fundamental_matrix, inliers, pts1, pts2


############## Find good keypoints to use ##############
kp1, des1, kp2, des2, flann_match_pairs = get_keypoints_and_descriptors(imgL, imgR)
#good_matches = lowes_ratio_test(flann_match_pairs, 0.2)
draw_matches(imgL, imgR, kp1, des1, kp2, des2, flann_match_pairs)


############## Compute Fundamental Matrix ##############
F, I, points1, points2 = compute_fundamental_matrix(good_matches, kp1, kp2)


############## Stereo rectify uncalibrated ##############
h1, w1 = imgL.shape[0:2]
h2, w2 = imgR.shape[0:2]

print(h1,w1,h2,w2)

points1 = points1[I.ravel()==1]
points2 = points2[I.ravel()==1]
thresh = 0
_, H1, H2 = cv2.stereoRectifyUncalibrated(
    np.float32(points1), np.float32(points2), F, imgSize=(w1, h1), threshold=thresh,
)

############## Undistort (Rectify) ##############
imgL_undistorted = cv2.warpPerspective(imgL, H1, (w1, h1))
imgR_undistorted = cv2.warpPerspective(imgR, H2, (w2, h2))
cv2.imwrite("undistorted_L.png", imgL_undistorted)
cv2.imwrite("undistorted_R.png", imgR_undistorted)

############## Calculate Disparity (Depth Map) ##############

# Using StereoBM
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity_BM = stereo.compute(imgL_undistorted, imgR_undistorted)
plt.imshow(disparity_BM, "gray")
plt.colorbar()
plt.show()

# Using StereoSGBM
# Set disparity parameters. Note: disparity range is tuned according to
#  specific parameters obtained through trial and error.
win_size = 3
min_disp = -4
max_disp = 9
num_disp = max_disp - min_disp  # Needs to be divisible by 16
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=5,
    uniquenessRatio=5,
    speckleWindowSize=5,
    speckleRange=5,
    disp12MaxDiff=2,
    P1=8 * 3 * win_size ** 2,
    P2=32 * 3 * win_size ** 2,
)
disparity_SGBM = stereo.compute(imgL_undistorted, imgR_undistorted)
plt.imshow(disparity_SGBM, "gray")
plt.colorbar()
plt.show()
