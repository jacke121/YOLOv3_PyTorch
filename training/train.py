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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from tensorboardX import SummaryWriter

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'evaluate'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from common.coco_dataset import COCODataset

checkpoint_dir="checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
def train(config):
    config["global_step"] = config.get("start_step", 0)
    is_training = False if config.get("export_onnx") else True

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

    # Optimizer and learning rate
    optimizer = _get_optimizer(config, net)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr"]["decay_step"],
        gamma=config["lr"]["decay_gamma"])

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLayer(config["batch_size"],i,config["yolo"]["anchors"][i],
                                     config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # DataLoader
    dataloader = torch.utils.data.DataLoader(COCODataset(config["train_path"],
                                                         (config["img_w"], config["img_h"]),
                                                         is_training=True,is_scene=False),
                                             batch_size=config["batch_size"],
                                             shuffle=True,drop_last=True, num_workers=0, pin_memory=True)

    # Start the training loop
    logging.info("Start training.")
    dataload_len=len(dataloader)
    for epoch in range(config["epochs"]):
        recall = 0
        mini_step = 0
        for step, samples in enumerate(dataloader):
            images, labels = samples["image"], samples["label"]
            start_time = time.time()
            config["global_step"] += 1
            for mini_batch in range(6):
                mini_step += 1
                # Forward and backward
                optimizer.zero_grad()
                outputs = net(images)
                losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls", "recall"]
                losses = [0] * len(losses_name)
                for i in range(3):
                    _loss_item = yolo_losses[i](outputs[i], labels)
                    for j, l in enumerate(_loss_item):
                        losses[j] += l
                # losses = [sum(l) for l in losses]
                loss = losses[0]
                loss.backward()
                optimizer.step()
                _loss = loss.item()
                # example_per_second = config["batch_size"] / duration
                # lr = optimizer.param_groups[0]['lr']

                strftime = datetime.datetime.now().strftime("%H:%M:%S")
                if (losses[7] / 3 > recall / (step+1)) or mini_batch == 5:
                    recall += losses[7] / 3
                    print(
                        '%s [Epoch %d/%d,batch %03d/%d loss:x %.5f,y %.5f,w %.5f,h %.5f,conf %.5f,cls %.5f,total %.5f,rec %.3f,avrec %.3f %d]' %
                        (strftime, epoch, config["epochs"], step, dataload_len,
                         losses[1], losses[2], losses[3],
                         losses[4], losses[5], losses[6],
                         _loss, losses[7] / 3, recall / (step+1), mini_batch))
                    break
                else:
                    print(
                        '%s [Epoch %d/%d,batch %03d/%d loss:x %.5f,y %.5f,w %.5f,h %.5f,conf %.5f,cls %.5f,total %.5f,rec %.3f,prerc %.3f %d]' %
                        (strftime, epoch, config["epochs"], step, dataload_len,
                         losses[1], losses[2], losses[3],
                         losses[4], losses[5], losses[6],
                         _loss, losses[7] / 3, recall / step, mini_batch))
        if (epoch % 2 == 0 and recall / len(dataloader) > 0.7) or recall / len(dataloader) > 0.96:
            torch.save(net.state_dict(), '%s/%.4f_%04d.weights' % (checkpoint_dir, recall / len(dataloader), epoch))

        lr_scheduler.step()
    # net.train(True)
    logging.info("Bye bye")

def _get_optimizer(config, net):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, net.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    else:
        logging.info("freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
        ]

    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        # Default to sgd
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")


    params_path = 'params.py'
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Create sub_working_dir
    sub_working_dir = '{}/{}/size{}x{}_try{}/{}'.format(
        config['working_dir'], config['model_params']['backbone_name'], 
        config['img_w'], config['img_h'], config['try'],
        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    # Creat tf_summary writer
    # config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    # logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    train(config)

