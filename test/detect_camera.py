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


import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from common.utils import non_max_suppression, bbox_iou
from numpy import unicode

ttfont = ImageFont.truetype("e:/tool/msyh.ttf", 15)
TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326",
        "classes": 80,
    },
    "batch_size": 1,
    "confidence_threshold": 0.8,
    "classes_names_path": "../data/coco2cls.names",
    "iou_thres": 0.3,
    "val_path": r"D:\data\VOC2007",
    "images_path":  r"\\192.168.55.39\team-CV\dataset\tiny_data_0627\train\_2test\JPEGImages/",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
}

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        # self.label = label
        self.score = -1

def draw_boxes(image, boxes, labels):
    for box in boxes:
        cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 2)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        # draw.text((box.xmin, box.ymin - 13), unicode(labels[box.classes] + " " + str(box.objness)), fill=(255, 0, 0),
        #           font=ttfont)
        draw.text((box.xmin, box.ymin - 13), unicode(labels[box.classes]), fill=(255, 0, 0), font=ttfont)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image

labels = ["人体", "自行车", "汽车", "摩托车", "飞机", "公共汽车", "火车", "卡车", \
          "船", "交通灯", "消防栓", "停止标识", "停车收费表", "长凳", \
          "鸟", "猫", "狗", "马", "羊", "牛", "象", "熊", "斑马", "长颈鹿", \
          "背包", "伞", "手提包", "领带", "行李箱", "飞盘", "滑雪板", "滑雪板", \
          "运动球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", \
          "网球拍", "瓶子", "酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", \
          "苹果", "三明治", "橙子", "花椰菜", "胡萝卜", "热狗", "比萨饼", "甜甜圈", "蛋糕", \
          "椅子", "沙发", "盆栽", "床", "餐桌", "马桶", "显示器", "笔记本电脑", "鼠标", \
          "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机", "水槽", "冰箱", \
          "书", "钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"]

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

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./007.avi")

    img_i = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_i += 1
        # preprocess
        images = []
        images_origin = []
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            time1=datetime.datetime.now()
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            print("time1",(datetime.datetime.now()-time1).microseconds)
            batch_detections = non_max_suppression(output, config["yolo"]["classes"],
                                                   conf_thres=config["confidence_threshold"])
            print("time2", (datetime.datetime.now() - time1).microseconds)

        # write result images. Draw bounding boxes and labels of detections
        classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]
        if not os.path.isdir("./output/"):
            os.makedirs("./output/")
        for idx, detections in enumerate(batch_detections):
            img_show = images_origin[idx]
            img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
            if detections is not None:
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                boxes=[]
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # Rescale coordinates to original dimensions
                    ori_h, ori_w = images_origin[idx].shape[:2]
                    pre_h, pre_w = config["img_h"], config["img_w"]
                    box_h = ((y2 - y1) / pre_h) * ori_h
                    box_w = ((x2 - x1) / pre_w) * ori_w
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w
                    # Create a Rectangle patch
                    box = BoundBox(x1, y1, x1 + box_w, y1 + box_h, cls_conf.item(), int(cls_pred))
                    boxes.append(box)
                    img_show = draw_boxes(img_show, boxes, labels)
                    # image_show = cv2.rectangle(images_origin[idx], (x1, y1), (x1 + box_w, y1 + box_h), (0, 255, 0), 1)

            cv2.imshow('1', img_show)
            cv2.waitKey(1)
    logging.info("Save all results to ./output/")

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,format="[%(asctime)s %(filename)s] %(message)s")

    config =TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    test(config)
