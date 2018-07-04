# -*- coding:utf-8 -*-
from __future__ import division

from multiprocessing import Process, Queue, Lock, Manager
from multiprocessing.pool import Pool
from common.coco_dataset_single import COCODataset
import logging
import torch.nn as nn
import os
import sys
import time
import datetime
import argparse
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from training import params

import torch
from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch.optim as optim


checkpoint_dir="checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
config = params.TRAINING_PARAMS
config["batch_size"] *= len(config["parallels"])

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
def get_data(queue,lock):
        dataloader = DataLoader(COCODataset(config["train_path"],
                                                         (config["img_w"], config["img_h"]),
                                                         is_training=True, data_size=100000),
        batch_size=config["batch_size"], shuffle=True,drop_last=True, num_workers=0)
        queue.put(("len",len(dataloader)))
        while True:
            for batch_i, samples in enumerate(dataloader):
                is_put=1
                while is_put:
                    lock.acquire()
                    if queue.qsize()<1:
                        queue.put(samples)
                        lock.release()
                        is_put=0
                    else:
                        time.sleep(0.005)
                        lock.release()

def train(config):
    config["global_step"] = config.get("start_step", 0)
    is_training = False if config.get("export_onnx") else True

    anchors = [int(x) for x in config["yolo"]["anchors"].split(",")]
    anchors = [[[anchors[i], anchors[i + 1]], [anchors[i + 2], anchors[i + 3]], [anchors[i + 4], anchors[i + 5]]]
               for i
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
        yolo_losses.append(YOLOLayer(config["batch_size"], i, config["yolo"]["anchors"][i],
                                     config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    total_loss = 0
    last_total_loss = 0

    manager = Manager()
    # 父进程创建Queue，并传给各个子进程：
    q = manager.Queue(1)
    lock = manager.Lock()  # 初始化一把锁
    p = Pool()
    pw = p.apply_async(get_data, args=(q, lock))


    batch_len=q.get()
    if batch_len[0]=="len":
        batch_len=batch_len[1]
    logging.info("Start training.")
    for epoch in range(config["epochs"]):
        recall = 0
        for step in range(batch_len):
            samples = q.get()
            images, labels = samples["image"], samples["label"]
            start_time = time.time()
            config["global_step"] += 1

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

            if step > 0 and step % 2 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config["batch_size"] / duration
                lr = optimizer.param_groups[0]['lr']

                strftime = datetime.datetime.now().strftime("%H:%M:%S")
                recall += losses[7] / 3
                print(
                    '%s [Epoch %d/%d, Batch %03d/%d losses: x %.5f, y %.5f, w %.5f, h %.5f, conf %.5f, cls %.5f, total %.5f, recall: %.3f]' %
                    (strftime, epoch, config["epochs"], step, batch_len,
                     losses[1], losses[2], losses[3],
                     losses[4], losses[5], losses[6],
                     _loss, losses[7] / 3))
                # logging.info(epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f "%
                #     (epoch, step, _loss, example_per_second, lr))
                # config["tensorboard_writer"].add_scalar("lr",
                #                                         lr,
                #                                         config["global_step"])
                # config["tensorboard_writer"].add_scalar("example/sec",
                #                                         example_per_second,
                #                                         config["global_step"])
                # for i, name in enumerate(losses_name):
                #     value = _loss if i == 0 else losses[i]
                #     config["tensorboard_writer"].add_scalar(name,
                #                                             value,
                #                                             config["global_step"])

        if (epoch % 2 == 0 and recall / batch_len > 0.7) or recall / batch_len > 0.96:
            torch.save(net.state_dict(), '%s/%04d.weights' % (checkpoint_dir, epoch))

        lr_scheduler.step()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

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


