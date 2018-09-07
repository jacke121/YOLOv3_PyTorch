import datetime
import os
import threading

import numpy as np
import logging
import cv2

import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from common import data_transforms
CLASSES = ('mouse',)
class MyThread(threading.Thread):
    def __init__(self,arg):
        super(MyThread, self).__init__()#注意：一定要显式的调用父类的初始化函数。
        self.listDataset=arg
    def run(self):#定义每个线程要运行的函数
      for index in range(len(self.listDataset.img_files)):
          if index in self.listDataset.img_d:
              pass
          else:
              img_path = self.listDataset.img_files[index % len(self.listDataset.img_files)].rstrip()
              img = cv2.imread(img_path, cv2.IMREAD_COLOR)
              if img is None:
                  raise Exception("Read image error: {}".format(img_path))
              img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              h, w, c = img.shape
              label_path = self.listDataset.label_files[index % len(self.listDataset.img_files)].rstrip()
              if os.path.exists(label_path):
                  # labels = np.loadtxt(label_path).reshape(-1, 5)

                  tree = ET.parse(label_path)
                  objs = tree.findall('object')
                  # num_objs = len(objs)
                  # labels = np.zeros(shape=(num_objs, 5), dtype=np.float64)
                  labels = []
                  for ix, obj in enumerate(objs):
                      if obj.find('difficult') is None or str(obj.find('difficult').text) == "0":
                          bbox = obj.find('bndbox')
                          x1 = max(float(bbox.find('xmin').text), 1)  # - 1
                          y1 = max(float(bbox.find('ymin').text), 1)  # - 1
                          x2 = min(float(bbox.find('xmax').text), 1279)  # - 1
                          y2 = min(float(bbox.find('ymax').text), 719)  # - 1
                          cls = self.listDataset._class_to_ind[obj.find('name').text.lower().strip()]
                          label_ = [cls, ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h]
                          # labels[ix, :] = [cls, ((x1 + x2) / 2) / padded_w, ((y1 + y2) / 2) / padded_h,
                          #                  w / padded_w, h / padded_h]
                          # labels[ix, :] = [cls, ((x1 + x2) / 2) / padded_w, ((y1 + y2) / 2) / padded_h,
                          #                  w_new / padded_w, h_new / padded_h]
                          labels.append(label_)
                  labels = np.asarray(labels)
              else:
                  logging.info("label does not exist: {}".format(label_path))
                  labels = np.zeros((1, 5), np.float32)

              sample = {'image': img, 'label': labels,"img_path":img_path}
              if self.listDataset.transforms is not None:
                  sample = self.listDataset.transforms(sample)
              self.listDataset.img_d[index] = sample
      print("data_load ok")
class COCODataset(Dataset):
    def __init__(self, list_path, img_size, is_training, is_debug=False,data_size=4000,is_scene=False):

        if is_scene:
            all_files = [list(map(lambda x: os.path.join(root, x), files)) for root, _, files in
                         os.walk(list_path, topdown=False) if os.path.basename(root) == 'Annotations']
            self.label_files = []
            for i in range(len(all_files)):
                self.label_files += all_files[i]
            if len(self.label_files)>data_size:
                self.label_files=self.label_files[:data_size]
            self.img_files = [file.replace('Annotations', 'JPEGImages').replace('xml', 'jpg') for file in
                              self.label_files]
        else:
            list_path_txt = os.path.join(list_path, 'ImageSets\Main/trainval.txt')
            if not is_training:
                list_path_txt = os.path.join(list_path, 'ImageSets\Main/test.txt')
            with open(list_path_txt, 'r') as file:
                # with open(list_path, 'r') as file:
                self.train_files_ = file.readlines()
            if len(self.train_files_) > data_size:
                self.train_files_ = self.train_files_[:data_size]
            if len(self.label_files) > data_size:
                self.label_files = self.label_files[:data_size]
            self.img_files = [os.path.join(list_path, 'JPEGImages', '%s.jpg' % train_file.strip('\n')) for train_file in
                              self.train_files_]
            self.label_files = [os.path.join(list_path, 'Annotations', '%s.xml' % train_file.strip('\n')) for train_file
                                in
                                self.train_files_]

        self.img_size = img_size  # (w, h)
        self.max_objects = 10
        self.is_debug = is_debug

        #  transforms and augmentation
        self.transforms = data_transforms.Compose()
        # if is_training:
        #     self.transforms.add(data_transforms.ImageBaseAug())
        # self.transforms.add(data_transforms.KeepAspect())
        self.transforms.add(data_transforms.ResizeImage(self.img_size))
        self.transforms.add(data_transforms.ToTensor(self.max_objects, self.is_debug))

        self._class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))
        self.img_d = {}
        self.lock = threading.RLock()
        t = MyThread(self)
        t.start()
    def __getitem__(self, index):
        if index in self.img_d:
            return self.img_d[index]
        else:
            self.lock.acquire()
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("Read image error: {}".format(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h,w,c = img.shape
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            if os.path.exists(label_path):
                # labels = np.loadtxt(label_path).reshape(-1, 5)

                tree = ET.parse(label_path)
                objs = tree.findall('object')
                # num_objs = len(objs)
                # labels = np.zeros(shape=(num_objs, 5), dtype=np.float64)
                labels = []
                for ix, obj in enumerate(objs):
                    if obj.find('difficult') is None or str(obj.find('difficult').text) == "0":
                        bbox = obj.find('bndbox')
                        x1 = max(float(bbox.find('xmin').text), 1)  # - 1
                        y1 = max(float(bbox.find('ymin').text), 1)  # - 1
                        x2 = min(float(bbox.find('xmax').text), 1279)  # - 1
                        y2 = min(float(bbox.find('ymax').text), 719)  # - 1
                        cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                        label_ = [cls, ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h]
                        # labels[ix, :] = [cls, ((x1 + x2) / 2) / padded_w, ((y1 + y2) / 2) / padded_h,
                        #                  w / padded_w, h / padded_h]
                        # labels[ix, :] = [cls, ((x1 + x2) / 2) / padded_w, ((y1 + y2) / 2) / padded_h,
                        #                  w_new / padded_w, h_new / padded_h]
                        labels.append(label_)
                labels = np.asarray(labels)
            else:
                logging.info("label does not exist: {}".format(label_path))
                labels = np.zeros((1, 5), np.float32)

            sample = {'image': img, 'label': labels,"img_path":img_path}
            if self.transforms is not None:
                sample = self.transforms(sample)
            self.img_d[index] = sample
            self.lock.release()
            return sample

    def __len__(self):
        return len(self.img_files)


#  use for test dataloader
if __name__ == "__main__":

    tree = ET.parse(r"\\192.168.55.73\Team-CV\dataset\original_0706\Annotations\09981.xml")
    objs = tree.findall('object')
    dataloader = torch.utils.data.DataLoader(COCODataset(r"D:\data\tiny_data\VOC2007",
                                                         (416, 416),is_training=True, is_debug=True),
                                             batch_size=16,
                                             shuffle=False, num_workers=0, pin_memory=False)
    for step, sample in enumerate(dataloader):
        for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
            image = image.numpy()
            h, w = image.shape[:2]
            for l in label:
                if l.sum() == 0:
                    continue
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            print(datetime.datetime.now())
            # cv2.imwrite("step{}_{}.jpg".format(step, i), image)
        # only one batch
        # break

