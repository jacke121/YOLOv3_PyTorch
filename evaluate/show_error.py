# coding='utf-8'
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

import torch
import torch.nn as nn

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None,label=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = label
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def draw_boxes(image, boxes, labels, obj_thresh):
    for box in boxes:
        cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 1)
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # draw = ImageDraw.Draw(image)
        # draw.text((box.xmin, box.ymin - 13), unicode(labels[box.classes] + " " + str(box.objness)), fill=(255, 0, 0),
        #           font=ttfont)
        # draw.text((box.xmin, box.ymin - 13), unicode(labels[box.classes]), fill=(255, 0, 0), font=ttfont)
        # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from common.utils import non_max_suppression, bbox_iou

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

labels = ["mouse"]
def test(config):
    is_training = False
    anchors = [int(x) for x in config["yolo"]["anchors"].split(",")]
    anchors = [[[anchors[i], anchors[i + 1]], [anchors[i + 2], anchors[i + 3]], [anchors[i + 4], anchors[i + 5]]] for i
               in range(0, len(anchors), 6)]
    anchors.reverse()
    config["yolo"]["anchors"] = []
    for i in range(3):
        config["yolo"]["anchors"].append(anchors[i])
    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLayer(config["batch_size"],i,config["yolo"]["anchors"][i],
                                     config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # prepare images path
    images_name = os.listdir(config["images_path"])
    images_path = [os.path.join(config["images_path"], name) for name in images_name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["images_path"]))

    # Start inference
    batch_size = config["batch_size"]
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        images_origin = []
        for path in images_path[step*batch_size: (step+1)*batch_size]:
            logging.info("processing: {}".format(path))
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin.append(image)  # keep for save result
            image = cv2.resize(image, (config["img_w"], config["img_h"]),
                               interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        # inference
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, config["yolo"]["classes"],
                                                   conf_thres=config["confidence_threshold"])

        # write result images. Draw bounding boxes and labels of detections
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
                boxes = []
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Rescale coordinates to original dimensions
                    ori_h, ori_w = images_origin[idx].shape[:2]
                    pre_h, pre_w = config["img_h"], config["img_w"]
                    box_h = ((y2 - y1) / pre_h) * ori_h
                    box_w = ((x2 - x1) / pre_w) * ori_w
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w
                    # Create a Rectangle patch

                    box = BoundBox(x1, y1, x1 + box_w, y1 + box_h, cls_conf.item(), int(cls_pred),
                                   classes[int(cls_pred)])
                    boxes.append(box)
                    # bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1,
                    #                          edgecolor=color,
                    #                          facecolor='none')
                    # # Add the bbox to the plot
                    # ax.add_patch(bbox)
                    # Add label
                    # plt.text(x1, y1, s=classes[int(cls_pred)], color='white',
                    #          verticalalignment='top',
                    #          bbox={'color': color, 'pad': 0})
            # Save generated image with detections
            img_show = draw_boxes(images_origin[idx], boxes, labels,0.5)
            img_show = cv2.resize(img_show, (img_show.shape[1], img_show.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
            # outVideo.write(img_show)
            cv2.imshow("ai", img_show)
            cv2.waitKey()
            # plt.close()
    logging.info("Save all results to ./output/")

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,format="[%(asctime)s %(filename)s] %(message)s")

    params_path =r"params.py"
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    test(config)
