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
		
		
params = dict()
params["model_folder"] = "/home/ruthz/openpose/models/"
params["face"] = False
params["hand"] = False
params['net_resolution'] = "-1x272"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()