# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
sys.path.append('/home/ruthz/openpose/build/python')
from openpose import pyopenpose as op
import glob 
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

def DrawKeypoint(img, kps):
    for kp in kps:
        cv2.drawMarker(img, (kp[0], kp[1]),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2, line_type=cv2.LINE_AA)
    return img


params = dict()
params["model_folder"] = "/home/ruthz/openpose/models/"
params["face"] = False
params["hand"] = False
params['net_resolution'] = "-1x272"

params["heatmaps_add_parts"] = True
params["heatmaps_add_bkg"] = True
params["heatmaps_add_PAFs"] = True
params["heatmaps_scale"] = 2

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()


imgTop = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_top_stitched/bytes-cafe-2019-02-07_0/000000.jpg")  # left image
imgBottom = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_bottom_stitched/bytes-cafe-2019-02-07_0/000000.jpg")  # right image
truth_file = "/home/ruthz/Desktop/desktop/jrdb_eval/kitti_labels/bytes-cafe-2019-02-07_0/000000.txt"

def GetBBoxes(labels_text_file):
	bboxes = []
	for line in open(labels_text_file):
		words = line.split(" ")
		bbox = [int(float(words[5])), int(float(words[6])), int(float(words[7])),int(float(words[8]))]
		bboxes.append(bbox)
	return bboxes
    

def GetBBoxes3D(labels_text_file):
	bboxes = []
	for line in open(labels_text_file):
		words = line.split(" ")
		bbox = [float(words[10]), float(words[9]),float(words[11]), float(words[9]), float(words[10]),float(words[11]) ]
		bboxes.append(bbox)
	return bboxes  

def DrawSkeleton(x,y,z,ax):
    for idx in range(len(x)):
        i1 =0 ; i2 = 1 
        CheckAndPlot([x[i1],y[i1],z[i1]], [x[i2],y[i2],z[i2]] ,ax)
        i1 =2 ; i2 = 1
        CheckAndPlot([x[i1],y[i1],z[i1]], [x[i2],y[i2],z[i2]] ,ax)        
        i1 =2 ; i2 = 3
        CheckAndPlot([x[i1],y[i1],z[i1]], [x[i2],y[i2],z[i2]] ,ax)
        i1 =5 ; i2 = 1
        CheckAndPlot([x[i1],y[i1],z[i1]], [x[i2],y[i2],z[i2]] ,ax)
        i1 =5 ; i2 = 6
        CheckAndPlot([x[i1],y[i1],z[i1]], [x[i2],y[i2],z[i2]] ,ax)                        
        i1 =8 ; i2 = 1
        CheckAndPlot([x[i1],y[i1],z[i1]], [x[i2],y[i2],z[i2]] ,ax)
        i1 =8 ; i2 = 12
        CheckAndPlot([x[i1],y[i1],z[i1]], [x[i2],y[i2],z[i2]] ,ax)
        i1 =8 ; i2 = 9
        CheckAndPlot([x[i1],y[i1],z[i1]], [x[i2],y[i2],z[i2]] ,ax)                        
        i1 =10 ; i2 = 9
        CheckAndPlot([x[i1],y[i1],z[i1]], [x[i2],y[i2],z[i2]] ,ax)        
        i1 =13 ; i2 = 12
        CheckAndPlot([x[i1],y[i1],z[i1]], [x[i2],y[i2],z[i2]] ,ax)        
        """
        ax.plot3D([x[0],x[1]], [y[0],y[1]], [z[0],z[1]], color="blue") 
        ax.plot3D([x[2],x[1]], [y[2],y[1]], [z[2],z[1]], color="blue") 
        ax.plot3D([x[2],x[3]], [y[2],y[3]], [z[2],z[3]], color="blue") 
        ax.plot3D([x[5],x[1]], [y[5],y[1]], [z[5],z[1]], color="blue") 
        ax.plot3D([x[5],x[6]], [y[5],y[6]], [z[5],z[6]], color="blue") 
        ax.plot3D([x[8],x[1]], [y[8],y[1]], [z[8],z[1]], color="blue") 
        ax.plot3D([x[8],x[12]], [y[8],y[12]], [z[8],z[12]], color="blue") 
        ax.plot3D([x[8],x[9]], [y[8],y[9]], [z[8],z[9]], color="blue") 
        ax.plot3D([x[10],x[9]], [y[10],y[9]], [z[10],z[9]], color="blue") 
        ax.plot3D([x[13],x[12]], [y[13],y[12]], [z[13],z[12]], color="blue") 
        """
    return ax

def CheckAndPlot(p1,p2,ax):
    if not (not np.any(p1) or not np.any(p2)):
        ax.plot3D([p1[0], p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], color="blue" )
    return ax

def PlotBoxes(truth_file, ax):
    bboxes = GetBBoxes3D(truth_file)
    for bbox in bboxes:
        top_4 = []
        bottom_4 = []
        top_4.append([bbox[3] + bbox[0]/2, bbox[4] + bbox[1]/2, bbox[5] + bbox[2]/2])
        top_4.append([bbox[3] + bbox[0]/2, bbox[4] - bbox[1]/2, bbox[5] + bbox[2]/2])
        top_4.append([bbox[3] - bbox[0]/2, bbox[4] - bbox[1]/2, bbox[5] + bbox[2]/2])
        top_4.append([bbox[3] - bbox[0]/2, bbox[4] + bbox[1]/2, bbox[5] + bbox[2]/2])
        bottom_4.append([bbox[3] + bbox[0]/2, bbox[4] + bbox[1]/2, bbox[5] - bbox[2]/2])
        bottom_4.append([bbox[3] + bbox[0]/2, bbox[4] - bbox[1]/2, bbox[5] - bbox[2]/2])
        bottom_4.append([bbox[3] - bbox[0]/2, bbox[4] - bbox[1]/2, bbox[5] - bbox[2]/2])
        bottom_4.append([bbox[3] - bbox[0]/2, bbox[4] + bbox[1]/2, bbox[5] - bbox[2]/2])    
        ax.plot3D([top_4[0][0],top_4[1][0]], [top_4[0][1],top_4[1][1]], [top_4[0][2],top_4[1][2]], color="blue") 
        ax.plot3D([top_4[1][0],top_4[2][0]], [top_4[1][1],top_4[2][1]], [top_4[1][2],top_4[2][2]], color="blue") 
        ax.plot3D([top_4[2][0],top_4[3][0]], [top_4[2][1],top_4[3][1]], [top_4[2][2],top_4[3][2]], color="blue") 
        ax.plot3D([top_4[3][0],top_4[0][0]], [top_4[3][1],top_4[0][1]], [top_4[3][2],top_4[0][2]], color="blue")
        ax.plot3D([bottom_4[0][0],bottom_4[1][0]], [bottom_4[0][1],bottom_4[1][1]], [bottom_4[0][2],bottom_4[1][2]], color="blue") 
        ax.plot3D([bottom_4[1][0],bottom_4[2][0]], [bottom_4[1][1],bottom_4[2][1]], [bottom_4[1][2],bottom_4[2][2]], color="blue") 
        ax.plot3D([bottom_4[2][0],bottom_4[3][0]], [bottom_4[2][1],bottom_4[3][1]], [bottom_4[2][2],bottom_4[3][2]], color="blue") 
        ax.plot3D([bottom_4[3][0],bottom_4[0][0]], [bottom_4[3][1],bottom_4[0][1]], [bottom_4[3][2],bottom_4[0][2]], color="blue")
        ax.plot3D([top_4[0][0],bottom_4[0][0]], [top_4[0][1],bottom_4[0][1]], [top_4[0][2],bottom_4[0][2]], color="blue")
        ax.plot3D([top_4[1][0],bottom_4[1][0]], [top_4[1][1],bottom_4[1][1]], [top_4[1][2],bottom_4[1][2]], color="blue")
        ax.plot3D([top_4[2][0],bottom_4[2][0]], [top_4[2][1],bottom_4[2][1]], [top_4[2][2],bottom_4[2][2]], color="blue")
        ax.plot3D([top_4[3][0],bottom_4[3][0]], [top_4[3][1],bottom_4[3][1]], [top_4[3][2],bottom_4[3][2]], color="blue")
    return ax 


############################################# Forward pass through openpose ######################################3
datum.cvInputData = imgTop
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
imgTopDet = datum.cvOutputData
imgTopKp = datum.poseKeypoints

#imgTopDet = cv2.rotate(imgTop,cv2.cv2.ROTATE_90_CLOCKWISE)


datum.cvInputData = imgBottom
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
imgBottomDet = datum.cvOutputData
imgBottomKp = datum.poseKeypoints
#tmgBottomDet =  cv2.rotate(imgBottom, cv2.cv2.ROTATE_90_CLOCKWISE)

fig, axs = plt.subplots(2)
axs[0].imshow(imgTopDet)
axs[1].imshow(imgBottomDet)
plt.show()


################################## Calculate Depth ###########################################3
f_y = 482.924
f_x = 480.459
c_x = 209.383
c_y = 355.144 
r = 3360000
B = 120 



cost = np.zeros((len(imgTopKp), len(imgBottomKp)))
for Topidx in range(len(imgTopKp)):
    for Bottomidx in range(len(imgBottomKp)):
        cost[Topidx, Bottomidx] = sum(abs(imgTopKp[Topidx,:,0] - imgBottomKp[Bottomidx,:,0]))

### Pose detection matching across two cameras and visualization ###
row_idxs , col_idxs = linear_sum_assignment(cost)

for row_idx, col_idx in zip(row_idxs, col_idxs):
    new_img_top = DrawKeypoint(imgTop, imgTopKp[row_idx])##img top 1 #img bottom 0 
    new_img_bottom = DrawKeypoint(imgBottom, imgBottomKp[col_idx])##img top 1 #img bottom 0 
    imgTop = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_top_stitched/bytes-cafe-2019-02-07_0/000000.jpg")  # left image
    imgBottom = cv2.imread("/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_bottom_stitched/bytes-cafe-2019-02-07_0/000000.jpg")  # right image
    fig, axs = plt.subplots(2)
    axs[0].imshow(new_img_top)
    axs[1].imshow(new_img_bottom)
    plt.show()
    #plt.imshow(imgTop)
    #plt.show()

 
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

#PlotBoxes(truth_file, ax)

for top_idx, bottom_idx in zip(row_idxs, col_idxs):
    disparity = imgTopKp[top_idx,:,1] - imgBottomKp[bottom_idx,:,1]
    depth = disparity*0
    h, w = imgTop.shape[:2]
    for i in range(len(depth)):
        if(disparity[i]!=0):
            depth[i] = f_y*B/disparity[i]
    ##for each person 
    ###depth = depth/1000
    depth = abs(depth)/1000.0
    #print(depth)
    theta = imgTopKp[top_idx,:,0]*2*np.pi / w  
    x = depth*np.sin(theta)
    z = depth*np.cos(theta)
    #h = (imgTopKp[top_idx, :, 1] - c_y)/f_y
    y = -1*z*(imgTopKp[top_idx,:,1] - c_y)/(f_y * np.cos(theta))
    print(len(x))
    ax = DrawSkeleton(x,y,z,ax)
    #ax.scatter(x, y, z, color="g", s=100)
plt.show()

"""
theta = (imgTopKp[top_idx,:,0]/w)*2*np.pi - np.pi
z = depth*np.cos(theta)
x = depth*np.sin(theta)
y = z*-1*(imgTopKp[top_idx,:,1] - c_y)/(f_y * np.cos(theta))
"""
    



"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

# draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")
        print(*zip(s, e))

        ### ax.plot3D([1,-1],[1,-1], [1,-1], color="blue")
# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="r")

# draw a point
ax.scatter([0], [0], [0], color="g", s=100)

# draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

a = Arrow3D([0, 1], [0, 1], [0, 1], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k")
ax.add_artist(a)
plt.show()
"""



# Display Image
counter = 0
while 1:
    num_maps = heatmaps.shape[0]
    heatmap = heatmaps[counter, :, :].copy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", heatmap)
    key = cv2.waitKey(-1)
    if key == 27:
        break
    counter += 1
    counter = counter % num_maps