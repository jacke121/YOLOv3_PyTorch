# coding='utf-8'
from __future__ import absolute_import
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random
import xml.etree.ElementTree as ET
import matplotlib
import configparser
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import training.params
import torch
import torch.nn as nn

config = training.params.TESTING_PARAMS
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from common.utils import non_max_suppression, bbox_iou

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
def read_gt_boxes(jpg_path):
    xml_file = jpg_path.replace('JPEGImages', 'Annotations').replace('jpg', 'xml')
    tree = ET.parse(xml_file)
    objs = tree.findall('object')
    num_objs = len(objs)
    labels = np.zeros(shape=(num_objs, 5), dtype=np.float64)
    bbox_list = []
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        x1 = max(float(bbox.find('xmin').text), 1)  # - 1
        y1 = max(float(bbox.find('ymin').text), 1)  # - 1
        x2 = min(float(bbox.find('xmax').text), 1279)  # - 1
        y2 = min(float(bbox.find('ymax').text), 719)  # - 1
        bbox_list.append((x1, x2, y1, y2))
    return bbox_list

def test(config):
    is_training = False
    anchors = [int(x) for x in config["yolo"]["anchors"].split(",")]
    anchors = [[[anchors[i], anchors[i + 1]], [anchors[i + 2], anchors[i + 3]], [anchors[i + 4], anchors[i + 5]]] for i
               in range(0, len(anchors), 6)]
    anchors.reverse()
    config["yolo"]["anchors"] = []

    for i in range(3):
        config["yolo"]["anchors"].append(anchors[i])
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()
    ini_files = os.listdir(os.path.join(config['test_weights'], 'result'))

    for kkk,ini_file in enumerate(ini_files):
        ini_list_config = configparser.ConfigParser()
        config_file_path = os.path.join(config['test_weights'], 'result',ini_files[-kkk-1])
        ini_list_config.read(config_file_path)
        ini_session = ini_list_config.sections()
        accuracy = ini_list_config.items(ini_session[0])
        err_jpgfiles = ini_list_config.items(ini_session[1])
        weight_file = os.path.join(config['test_weights'],'%s.weights'%ini_files[-kkk-1].split('_')[0])
        pass
        if weight_file:                    # Restore pretrain model
            logging.info("load checkpoint from {}".format(weight_file))
            state_dict = torch.load(weight_file)
            net.load_state_dict(state_dict)
        else:
            raise Exception("missing pretrain_snapshot!!!")

        yolo_losses = []
        for i in range(3):
            yolo_losses.append(YOLOLayer(config["batch_size"], i, config["yolo"]["anchors"][i],
                                         config["yolo"]["classes"], (config["img_w"], config["img_h"])))
        # images_name = os.listdir(config["images_path"]) # prepare images path
        # images_path = [os.path.join(config["images_path"], name) for name in images_name]
        # if len(images_path) == 0:
        #     raise Exception("no image found in {}".format(config["images_path"]))
        # batch_size = config["batch_size"]# Start inference
        # for step in range(0, len(images_path), batch_size):


        for _jpg_images in err_jpgfiles:
            images = []# preprocess
            images_origin = []
            jpg_path = str(_jpg_images[1])
            logging.info("processing: {}".format(jpg_path))
            bbox_list = read_gt_boxes(jpg_path)

            image = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
            if image is None:
                logging.error("read path error: {}. skip it.".format(jpg_path))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin.append(image)  # keep for save result
            image = cv2.resize(image, (config["img_w"], config["img_h"]),interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image)

            images = np.asarray(images)
            images = torch.from_numpy(images).cuda()
            with torch.no_grad():# inference
                outputs = net(images)
                output_list = []
                for i in range(3):
                    output_list.append(yolo_losses[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, config["yolo"]["classes"],
                                                       conf_thres=config["confidence_threshold"])
            classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]
            if not os.path.isdir("./output/"):
                os.makedirs("./output/")
            for idx, detections in enumerate(batch_detections):
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.imshow(images_origin[idx])
                if detections is not None:
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    bbox_colors = random.sample(colors, n_cls_preds)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        ori_h, ori_w = images_origin[idx].shape[:2]# Rescale coordinates to original dimensions
                        pre_h, pre_w = config["img_h"], config["img_w"]
                        box_h = ((y2 - y1) / pre_h) * ori_h
                        box_w = ((x2 - x1) / pre_w) * ori_w
                        y1 = (y1 / pre_h) * ori_h
                        x1 = (x1 / pre_w) * ori_w
                        image_show = cv2.rectangle(images_origin[idx], (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 255),2)
                    for (x1, x2, y1, y2) in bbox_list:
                        [x1, x2, y1, y2] = map(int, [x1, x2, y1, y2])
                        cv2.rectangle(image_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('1', image_show)
                cv2.waitKey()

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s %(filename)s] %(message)s")


    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    test(config)
