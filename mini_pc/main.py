
import configparser

import sys
import os

import cv2

weights_path=r"\\192.168.55.38\Team-CV\checkpoints"


from multiprocessing import Process, Queue, Lock
import numpy as np
sys.path.append("..")
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir + "..")

class Camera:
    def read_default(self):
        self.Cam_list_config = configparser.ConfigParser()
        self.config_file_path = current_dir + '/cam_list.ini'
        self.Cam_list_config.read(self.config_file_path)
        cam_session = self.Cam_list_config.sections()
        self.cam_name_list = self.Cam_list_config.items(cam_session[0])
        self.cv_default_model = self.Cam_list_config.items(cam_session[1])
    def open_cam(self):
      for cam_path in  self.cam_name_list:
          cap = cv2.VideoCapture(cam_path[1])
          print(cap.isOpened())
          while cap.isOpened():
              success, frame = cap.read()
              cv2.imshow("frame", frame)
              cv2.waitKey(1)



cam=Camera()
cam.read_default()
cam.open_cam()

