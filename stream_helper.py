# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 02:03:07 2021
@author: GJU5KOR
"""

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
from detection import Detectors
import os

from motpy import Detection, MultiObjectTracker, NpImage, Box
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track


eyes_cascade_path = "data\haarcascades\haarcascade_eye_tree_eyeglasses.xml"
eyes_cascade_path1 = "data\haarcascades\haarcascade_eye.xml"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",  help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=700, help="minimum area size")
ap.add_argument("-f", "--cascade-face", default="data\haarcascades\haarcascade_frontalface_alt2.xml", help = "path to where the face cascade resides")
ap.add_argument("-e", "--cascade-eyes", default=eyes_cascade_path, help="path to where the eyse cascade resides")
ap.add_argument("-o", "--output", default="data\Hari", help= "path to output directory")
args = vars(ap.parse_args())

#cacadepath = "data\haarcascades\haarcascade_eye_tree_eyeglasses.xml"
#cacadepath = "data\haarcascades\haarcascade_frontalface_alt2.xml"
#cacadepath = "data\haarcascades\haarcascade_upperbody.xml"

detectors = Detectors(args.get("cascade_face"), args.get("cascade_eyes"), True)
total = 0

# prepare multi object tracker
model_spec = {'order_pos': 1, 'dim_pos': 2,
              'order_size': 0, 'dim_size': 2,
              'q_var_pos': 5000., 'r_var_pos': 0.1}

dt = 1 / 15.0  # assume 15 fps
tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])

while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    frame = vs.read()
    if frame is None:
        break

    original_img = frame.copy()
    frame = imutils.resize(frame, width=500, height=500)

    has_moved = detectors.motion_detector(frame)

    count_faces_eyes = detectors.haar_face_eyes_finder(frame)
    print(count_faces_eyes)
    print("Printing counted faces and eyes", count_faces_eyes[0], count_faces_eyes[1])
    # if has_moved:
        # if has_eyes or has_face:
    #         print("execute face detection here.")
    #
    # print("Moved frame check value :", has_moved)
    # print("Is Face Exists in the frame:", has_face)
    # print("Is Eyes Found in frame:", has_eyes)

    # time.sleep(1)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("k"):
        p = os.path.sep.join([args["output"], "{}.png".format(
            str(total).zfill(5))])
        cv2.imwrite(p, frame)
        total += 1


    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
