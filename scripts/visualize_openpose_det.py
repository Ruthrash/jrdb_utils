# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
sys.path.append('/home/ruthz/openpose/build/python');
from openpose import pyopenpose as op
import glob 

base_dir = '/media/ruthz/data/cvgl/group/jrdb/data/train_dataset/images/image_stitched'
#base_dir = '/media/ruthz/data/jrdb_test/jrdb_test/cvgl/group/jrdb/data/test_dataset/images/image_stitched'

det_base_dir = '/home/ruthz/Desktop/jrdb2d_dets'
gt_base_dir = '/home/ruthz/Desktop/jrdb_eval/kitti_labels'
save_path_base_dir = '/home/ruthz/Desktop/jrdb_det_images'
seq_folders = glob.glob(base_dir+'/*/')
seq_folders.sort()

def GetBBoxes(labels_text_file):
	bboxes = []
	for line in open(labels_text_file):
		words = line.split(" ")
		bbox = [int(float(words[5])), int(float(words[6])), int(float(words[7])),int(float(words[8]))]
		bboxes.append(bbox)
	return bboxes
	


for seq in seq_folders:
	image_file_names = glob.glob(seq+'/*.jpg')
	image_file_names.sort()
	folder_name = seq.split("/")
	folder_name = folder_name[len(folder_name)-2]
	output_folder = save_path_base_dir+"/"+folder_name
	os.mkdir(output_folder)
	for image_file_name in image_file_names:
		##Storing results
		new_img = cv2.imread(image_file_name)		
		image_name = image_file_name.split("/")
		image_name = image_name[len(image_name) - 1]
		image_name = image_name.split(".")
		image_name = image_name[0]###gets individual image name
		gt_file = gt_base_dir+"/"+folder_name+"/"+image_name+".txt"
		det_file = det_base_dir+"/"+folder_name+"/"+image_name+".txt"
		det_boxes = GetBBoxes(det_file)
		gt_boxes = GetBBoxes(gt_file)
		for gt_box in gt_boxes:
			if(gt_box[0] >=0):
				cv2.rectangle(new_img,(gt_box[0],gt_box[1]),(gt_box[2],gt_box[3]),(0,255,0),3)##green
		for det_box in det_boxes:
			cv2.rectangle(new_img,(det_box[0],det_box[1]),(det_box[2],det_box[3]),(255,0,0),3)##red
		
		#cv2.imshow("a",new_img)
		#cv2.waitKey(0)
		
		
		output_file = output_folder+"/"+image_name+".jpg"
		#print(output_file)
		cv2.imwrite(output_file, new_img)
			
		
		
