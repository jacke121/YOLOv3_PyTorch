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
from torch.utils.data import dataloader
import torch
import torch.nn as nn
import configparser

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from common.coco_dataset import COCODataset
from common.utils import non_max_suppression, bbox_iou

TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": "15,22, 24,38, 25,64, 27,82, 39,58, 44,38, 62,77, 70,131, 78,233",
        "classes": 1,
    },
    "batch_size": 16,
    "iou_thres": 0.5,
    "train_path": r"\\192.168.55.39\team-CV\dataset\wuding_yy",
    # "train_path": r"\\192.168.55.39\team-CV\dataset\original_07062",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    # "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
    "pretrain_snapshot": "",
    # "pretrain_snapshot": r"C:\Users\sbdya\Desktop\tmp/140.weights",
}
def build_yolov3(config):
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
    net = nn.DataParallel(net)
    net = net.cuda()
    yolo_losses = []# YOLO loss with 3 scales
    for i in range(3):
        yolo_losses.append(YOLOLayer(config["batch_size"], i, config["yolo"]["anchors"][i],
                                     config["yolo"]["classes"], (config["img_w"], config["img_h"])))
    return net,yolo_losses
def write_ini(ini_name,accuracy,error_list):
    _conf_file = configparser.ConfigParser()
    f2 = open(ini_name, 'w')
    f2.close()
    _conf_file.read(ini_name)
    _conf_file.add_section("Accuracy")  # 增加section
    _conf_file.set("Accuracy", "acc", str(accuracy))  # 给新增的section 写入option

    _conf_file.add_section("error_list")
    for k, file in enumerate(error_list):
        _conf_file.set("error_list", str(k), file)
    print('write to ini %s' % ini_name)
    _conf_file.write(open(ini_name, 'w'))
def evaluate(config):
    # checkpoint_paths = {'58': r'\\192.168.25.58\Team-CV\checkpoints\torch_yolov3'}
    checkpoint_paths = {'39': r'\\192.168.55.39\team-CV\dataset\yolov3_torch/'}
    # checkpoint_paths = {'68': r'E:\github\YOLOv3_PyTorch\evaluate\weights'}
    post_weights = {k: 0 for k in checkpoint_paths.keys()}
    weight_index = {k: 0 for k in checkpoint_paths.keys()}
    time_inter = 10
    dataloader = torch.utils.data.DataLoader(COCODataset(config["train_path"],
                                                         (config["img_w"], config["img_h"]),
                                                         is_training=False),
                                             batch_size=config["batch_size"],
                                             shuffle=False, num_workers=0, pin_memory=False,drop_last = True)  # DataLoader
    net, yolo_losses = build_yolov3(config)
    while 1:
        for key,checkpoint_path in checkpoint_paths.items():
            os.makedirs(checkpoint_path + '/result', exist_ok=True)
            checkpoint_weights = os.listdir(checkpoint_path)
            checkpoint_result = os.listdir(checkpoint_path + '/result')
            checkpoint_result = [cweight.split("_")[1][:-4] for cweight in checkpoint_result if cweight.endswith('ini')]
            checkpoint_weights =[cweight for cweight in checkpoint_weights if cweight.endswith('weights')]

            if weight_index[key]>=len(checkpoint_weights):
                print('weight_index[key]',weight_index[key],len(checkpoint_weights))
                time.sleep(time_inter)
                continue
            if post_weights[key] == checkpoint_weights[weight_index[key]]:
                print('post_weights[key]', post_weights[key])
                time.sleep(time_inter)
                continue
            post_weights[key] = checkpoint_weights[weight_index[key]]

            if post_weights[key].endswith("_.weights"):#检查权重是否保存完
                print("post_weights[key].split('_')",post_weights[key].split('_'))
                time.sleep(time_inter)
                continue
            if checkpoint_weights[weight_index[key]].split("_")[1][:-8] in checkpoint_result:
                print('weight_index[key] +',weight_index[key])
                weight_index[key] += 1
                time.sleep(time_inter//20)
                continue
            weight_index[key] += 1
            try:
                if config["pretrain_snapshot"]:  # Restore pretrain model
                    state_dict = torch.load(config["pretrain_snapshot"])
                    logging.info("loading model from %s"%config["pretrain_snapshot"])
                    net.load_state_dict(state_dict)
                else:
                    state_dict = torch.load(os.path.join(checkpoint_path, post_weights[key]))
                    logging.info("loading model from %s" % os.path.join(checkpoint_path, post_weights[key]))
                    net.load_state_dict(state_dict)
            except Exception as E:
                print(E)
                time.sleep(time_inter)
                continue
            logging.info("Start eval.")# Start the eval loop
            n_gt = 0
            correct = 0
            imagepath_list = []
            for step, samples in enumerate(dataloader):
                images, labels ,image_paths= samples["image"], samples["label"],samples["img_path"]
                labels = labels.cuda()
                with torch.no_grad():
                    time1=datetime.datetime.now()
                    outputs = net(images)
                    print("time1",(datetime.datetime.now()-time1).microseconds)
                    output_list = []
                    for i in range(3):
                        output_list.append(yolo_losses[i](outputs[i]))
                    output = torch.cat(output_list, 1)
                    output = non_max_suppression(output, 1, conf_thres=0.8)
                    print("time2", (datetime.datetime.now() - time1).microseconds)
                    #  calculate
                    for sample_i in range(labels.size(0)):
                        # Get labels for sample where width is not zero (dummies)
                        target_sample = labels[sample_i, labels[sample_i, :, 3] != 0]
                        for obj_cls, tx, ty, tw, th in target_sample:
                            # Get rescaled gt coordinates
                            tx1, tx2 = config["img_w"] * (tx - tw / 2), config["img_w"] * (tx + tw / 2)
                            ty1, ty2 = config["img_h"] * (ty - th / 2), config["img_h"] * (ty + th / 2)
                            n_gt += 1
                            box_gt = torch.cat([coord.unsqueeze(0) for coord in [tx1, ty1, tx2, ty2]]).view(1, -1)
                            sample_pred = output[sample_i]
                            if sample_pred is not None:
                                # Iterate through predictions where the class predicted is same as gt
                                for x1, y1, x2, y2, conf, obj_conf, obj_pred in sample_pred[sample_pred[:, 6] == obj_cls]:
                                    box_pred = torch.cat([coord.unsqueeze(0) for coord in [x1, y1, x2, y2]]).view(1, -1)
                                    iou = bbox_iou(box_pred, box_gt)
                                    if iou >= config["iou_thres"]:
                                        correct += 1
                                        break
                                    else:
                                        if image_paths[sample_i] not in imagepath_list:
                                            imagepath_list.append(image_paths[sample_i])
                if n_gt:
                    logging.info('Batch [%d/%d] mAP: %.5f' % (step, len(dataloader), float(correct / n_gt)))

            logging.info('Mean Average Precision: %.5f' % float(correct / n_gt))
            Mean_Average = float(correct / n_gt)
            ini_name = os.path.join(checkpoint_path+'/result/', '%.4f_%s.ini'%(float(correct / n_gt),post_weights[key].split('.')[1].split("_")[1]))
            write_ini(ini_name, Mean_Average, imagepath_list)

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    config =TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    evaluate(config)

if __name__ == "__main__":
    main()
