import datetime

import csk
import numpy as np
from scipy.misc import imread, imsave
import cv2 # (Optional) OpenCV for drawing bounding boxes

length = 472 # sequence length

# 1st frame's groundtruth information
x1 = 255 # position x of the top-left corner of bounding box
y1 = 402 # position y of the top-left corner of bounding box
width = 74 # the width of bounding box
height = 55 # the height of bounding box

sequence_path = "" # your sequence path
save_path = "" # your save path

tracker = csk.CSK() # CSK instance
i=0

vi=cv2.VideoCapture(0)
while True:

    ret,frame=vi.read()
    # frame = imread(sequence_path+"%04d.jpg"%i)
    i+=1
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if i == 1: # 1st frame
        print(str(i)) # progress
        tracker.init(frame,x1,y1,width,height) # initialize CSK tracker with GT bounding box
        # imsave(save_path+'%04d.jpg'%i,cv2.rectangle(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), (x1, y1), (x1+width, y1+height), (0,255,0), 2)) # draw bounding box and save the frame

    else: # other frames
        # print(str(i)) # progress
        time1=datetime.datetime.now()
        x1, y1 = tracker.update(frame) # update CSK tracker and output estimated position
        print("time",(datetime.datetime.now()-time1).microseconds)
        imshow=cv2.rectangle(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), (x1, y1), (x1+width, y1+height), (0,255,0), 2)
        cv2.imshow("imshow",imshow)
        cv2.waitKey(1)
        # imsave(save_path+'%04d.jpg'%i,imshow) # draw bounding box and save the frame