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

from nets.mobilenet import MobileNet2

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from common.coco_dataset import COCODataset
from common.utils import non_max_suppression, bbox_iou
from datetime import timedelta
# from tensorboardX import SummaryWriter

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'evaluate'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from common.coco_dataset import COCODataset

checkpoint_dir=r"E:\Team-CV\mobile_0803"
os.makedirs(checkpoint_dir, exist_ok=True)

config ={
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained":"", #  set empty to disable
        # "backbone_pretrained":"../weights/mobilenetv2_weights.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": "16,24, 23,39, 25,84, 31,66, 42,54, 46,38, 56,81, 59,121, 74,236",
        "classes": 1,
    },
    "iou_thres":0.5,
    "lr": {
        "backbone_lr": 0.01,
        "other_lr": 0.01,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.4,
        "decay_step": 10,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "batch_size": 6,
    "train_path": r"\\192.168.55.73\team-CV\dataset\origin_all_datas\_2train",
    "epochs": 2001,
    "img_h": 416,
    "img_w": 416,
    # "parallels": [0,1,2,3],                         #  config GPU device
    "parallels": [0],                         #  config GPU device
    "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir
    # "pretrain_snapshot": "",                        #  load checkpoint
    "pretrain_snapshot": r"E:\Team-CV\YOLOv3_PyTorch\weights\0.8815_0438.weights",
    "evaluate_type": "",
    "try": 0,
    "export_onnx": False,
}
pruned_book = {}
weight_masks = []
bias_masks = []
index = 0
num_pruned = 0
num_weights = 0
def prune(m):
    global index, pruned_book, num_pruned
    global num_weights
    global weight_masks, bias_masks

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        num = torch.numel(m.weight.data)

        if type(m) == nn.Conv2d and index == 0:
            alpha = 0.2
        else:
            alpha = 1

        # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
        weight_mask = torch.ge(m.weight.data.abs(), alpha * m.weight.data.std()).type('torch.FloatTensor')

        if m.bias is not None:
            bias_mask = torch.ones(m.bias.data.size())
        weight_mask = weight_mask.cuda()
        if m.bias is not None:
            bias_mask = bias_mask.cuda()

        if len(weight_masks) <= index:
            weight_masks.append(weight_mask)
        else:
            weight_masks[index] = weight_mask
        # for all kernels in the conv2d layer, if any kernel is all 0, set the bias to 0
        # in the case of linear layers, we search instead for zero rows
        if m.bias is not None:
            for i in range(bias_mask.size(0)):
                if len(torch.nonzero(weight_mask[i]).size()) == 0:
                    bias_mask[i] = 0

            if len(bias_masks) <= index:
                bias_masks.append(bias_mask)
            else:
                bias_masks[index] = bias_mask

        index += 1
        layer_pruned = num - torch.nonzero(weight_mask).size(0)
        print('number pruned in weight of layer %d: %.3f %%' % (index, 100 * (layer_pruned / num)))
        if m.bias is not None:
            bias_num = torch.numel(bias_mask)
            bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
            print('number pruned in  bias of layer %d: %.3f %%' % (index, 100 * (bias_pruned / bias_num)))

        if index not in pruned_book.keys():
            pruned_book[index] = [100 * layer_pruned / num]
        else:
            pruned_book[index].append(100 * layer_pruned / num)

        num_pruned += layer_pruned
        num_weights += num

        m.weight.data *= weight_mask
        if m.bias is not None:
            m.bias.data *= bias_mask

    elif isinstance(m, ModelMain):
        for k, v in m._modules.items():
            for ck, c_m in v._modules.items():
                if isinstance(c_m, torch.nn.Conv2d):
                    num = torch.numel(c_m.weight.data)
                    alpha = 1

                    # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
                    weight_mask = torch.ge(c_m.weight.data.abs(), alpha * c_m.weight.data.std()).type(
                        'torch.FloatTensor')

                    if c_m.bias is not None:
                        bias_mask = torch.ones(c_m.bias.data.size())
                    weight_mask = weight_mask.cuda()
                    if c_m.bias is not None:
                        bias_mask = bias_mask.cuda()

                    if len(weight_masks) <= index:
                        weight_masks.append(weight_mask)
                    else:
                        weight_masks[index] = weight_mask
                    # for all kernels in the conv2d layer, if any kernel is all 0, set the bias to 0
                    # in the case of linear layers, we search instead for zero rows
                    if c_m.bias is not None:
                        for i in range(bias_mask.size(0)):
                            if len(torch.nonzero(weight_mask[i]).size()) == 0:
                                bias_mask[i] = 0

                        if len(bias_masks) <= index:
                            bias_masks.append(bias_mask)
                        else:
                            bias_masks[index] = bias_mask

                    index += 1
                    layer_pruned = num - torch.nonzero(weight_mask).size(0)
                    print('number pruned in weight of layer %d: %.3f %%' % (index, 100 * (layer_pruned / num)))
                    if c_m.bias is not None:
                        bias_num = torch.numel(bias_mask)
                        bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
                        print(
                            'number pruned in  bias of layer %d: %.3f %%' % (index, 100 * (bias_pruned / bias_num)))

                    if index not in pruned_book.keys():
                        pruned_book[index] = [100 * layer_pruned / num]
                    else:
                        pruned_book[index].append(100 * layer_pruned / num)

                    num_pruned += layer_pruned
                    num_weights += num

                    c_m.weight.data *= weight_mask
                    if c_m.bias is not None:
                        c_m.bias.data *= bias_mask

    elif isinstance(m, MobileNet2):
        for k, c_m in m._modules.items():
            if isinstance(c_m, torch.nn.Conv2d):
                num = torch.numel(c_m.weight.data)
                alpha = 1

                # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
                weight_mask = torch.ge(c_m.weight.data.abs(), alpha * c_m.weight.data.std()).type(
                    'torch.FloatTensor')

                if c_m.bias is not None:
                    bias_mask = torch.ones(c_m.bias.data.size())
                weight_mask = weight_mask.cuda()
                if c_m.bias is not None:
                    bias_mask = bias_mask.cuda()

                if len(weight_masks) <= index:
                    weight_masks.append(weight_mask)
                else:
                    weight_masks[index] = weight_mask
                # for all kernels in the conv2d layer, if any kernel is all 0, set the bias to 0
                # in the case of linear layers, we search instead for zero rows
                if c_m.bias is not None:
                    for i in range(bias_mask.size(0)):
                        if len(torch.nonzero(weight_mask[i]).size()) == 0:
                            bias_mask[i] = 0

                    if len(bias_masks) <= index:
                        bias_masks.append(bias_mask)
                    else:
                        bias_masks[index] = bias_mask

                index += 1
                layer_pruned = num - torch.nonzero(weight_mask).size(0)
                print('number pruned in weight of layer %d: %.3f %%' % (index, 100 * (layer_pruned / num)))
                if c_m.bias is not None:
                    bias_num = torch.numel(bias_mask)
                    bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
                    print(
                        'number pruned in  bias of layer %d: %.3f %%' % (index, 100 * (bias_pruned / bias_num)))

                if index not in pruned_book.keys():
                    pruned_book[index] = [100 * layer_pruned / num]
                else:
                    pruned_book[index].append(100 * layer_pruned / num)

                num_pruned += layer_pruned
                num_weights += num

                c_m.weight.data *= weight_mask
                if c_m.bias is not None:
                    c_m.bias.data *= bias_mask
    elif isinstance(m, torch.nn.ModuleList):
            pass
            for chikdmodule in m:
                for name, module in list(chikdmodule._modules.items()):
                    if isinstance(module, torch.nn.Conv2d):
                        num = torch.numel(module.weight.data)
                        alpha = 1

                        # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
                        weight_mask = torch.ge(module.weight.data.abs(), alpha * module.weight.data.std()).type(
                            'torch.FloatTensor')

                        if module.bias is not None:
                            bias_mask = torch.ones(module.bias.data.size())
                        weight_mask = weight_mask.cuda()
                        if module.bias is not None:
                            bias_mask = bias_mask.cuda()

                        if len(weight_masks) <= index:
                            weight_masks.append(weight_mask)
                        else:
                            weight_masks[index] = weight_mask
                        # for all kernels in the conv2d layer, if any kernel is all 0, set the bias to 0
                        # in the case of linear layers, we search instead for zero rows
                        if module.bias is not None:
                            for i in range(bias_mask.size(0)):
                                if len(torch.nonzero(weight_mask[i]).size()) == 0:
                                    bias_mask[i] = 0

                            if len(bias_masks) <= index:
                                bias_masks.append(bias_mask)
                            else:
                                bias_masks[index] = bias_mask

                        index += 1
                        layer_pruned = num - torch.nonzero(weight_mask).size(0)
                        print('number pruned in weight of layer %d: %.3f %%' % (index, 100 * (layer_pruned / num)))
                        if module.bias is not None:
                            bias_num = torch.numel(bias_mask)
                            bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
                            print(
                                'number pruned in  bias of layer %d: %.3f %%' % (index, 100 * (bias_pruned / bias_num)))

                        if index not in pruned_book.keys():
                            pruned_book[index] = [100 * layer_pruned / num]
                        else:
                            pruned_book[index].append(100 * layer_pruned / num)

                        num_pruned += layer_pruned
                        num_weights += num

                        module.weight.data *= weight_mask
                        if module.bias is not None:
                            module.bias.data *= bias_mask
    elif  isinstance(m, nn.Sequential):
        for k, v in m._modules.items():
            for ck,c_m in v._modules.items():
                if isinstance(c_m, torch.nn.Conv2d):
                    num = torch.numel(c_m.weight.data)
                    alpha = 1

                    # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
                    weight_mask = torch.ge(c_m.weight.data.abs(), alpha * c_m.weight.data.std()).type(
                        'torch.FloatTensor')

                    if c_m.bias is not None:
                        bias_mask = torch.ones(c_m.bias.data.size())
                    weight_mask = weight_mask.cuda()
                    if c_m.bias is not None:
                        bias_mask = bias_mask.cuda()

                    if len(weight_masks) <= index:
                        weight_masks.append(weight_mask)
                    else:
                        weight_masks[index] = weight_mask
                    # for all kernels in the conv2d layer, if any kernel is all 0, set the bias to 0
                    # in the case of linear layers, we search instead for zero rows
                    if c_m.bias is not None:
                        for i in range(bias_mask.size(0)):
                            if len(torch.nonzero(weight_mask[i]).size()) == 0:
                                bias_mask[i] = 0

                        if len(bias_masks) <= index:
                            bias_masks.append(bias_mask)
                        else:
                            bias_masks[index] = bias_mask

                    index += 1
                    layer_pruned = num - torch.nonzero(weight_mask).size(0)
                    print('number pruned in weight of layer %d: %.3f %%' % (index, 100 * (layer_pruned / num)))
                    if c_m.bias is not None:
                        bias_num = torch.numel(bias_mask)
                        bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
                        print(
                            'number pruned in  bias of layer %d: %.3f %%' % (index, 100 * (bias_pruned / bias_num)))

                    if index not in pruned_book.keys():
                        pruned_book[index] = [100 * layer_pruned / num]
                    else:
                        pruned_book[index].append(100 * layer_pruned / num)

                    num_pruned += layer_pruned
                    num_weights += num

                    c_m.weight.data *= weight_mask
                    if c_m.bias is not None:
                        c_m.bias.data *= bias_mask
    else:
        print("m",type(m))



def set_grad(m):
    global index
    global weight_masks, bias_masks
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        m.weight.grad.data *= weight_masks[index]
        if m.bias is not None and bias_masks is not None and index<len(bias_masks):
            m.bias.grad.data *= bias_masks[index]
        index += 1
    elif isinstance(m, torch.nn.ModuleList):
        for chikdmodule in m:
            for name, module in list(chikdmodule._modules.items()):
                if isinstance(module, torch.nn.Conv2d):
                    module.weight.grad.data *= weight_masks[index]
                    if module.bias is not None and bias_masks is not None and index < len(bias_masks):
                        module.bias.grad.data *= bias_masks[index]
                    index += 1
    elif isinstance(m, nn.Sequential):
        for k, v in m._modules.items():
            for ck, c_m in v._modules.items():
                if isinstance(c_m, torch.nn.Conv2d):
                    c_m.weight.grad.data *= weight_masks[index]
                    if c_m.bias is not None and bias_masks is not None and index < len(bias_masks):
                        c_m.bias.grad.data *= bias_masks[index]
                    index += 1
anchors = [int(x) for x in config["yolo"]["anchors"].split(",")]
anchors = [[[anchors[i], anchors[i + 1]], [anchors[i + 2], anchors[i + 3]], [anchors[i + 4], anchors[i + 5]]] for i
           in range(0, len(anchors), 6)]
anchors.reverse()
config["yolo"]["anchors"] = []
for i in range(3):
    config["yolo"]["anchors"].append(anchors[i])
yolo_losses = []
for i in range(3):
    yolo_losses.append(YOLOLayer(config["batch_size"],i,config["yolo"]["anchors"][i],
                                     config["yolo"]["classes"], (config["img_w"], config["img_h"])))

dataloader = torch.utils.data.DataLoader(COCODataset(config["train_path"],
                                                         (config["img_w"], config["img_h"]),
                                                         is_training=True,is_scene=True),
                                             batch_size=config["batch_size"],
                                             shuffle=True,drop_last=True, num_workers=0, pin_memory=True)
def validate(net):
    n_gt=0
    correct=0
    for step, samples in enumerate(dataloader):
        images, labels, image_paths = samples["image"], samples["label"], samples["img_path"]
        labels = labels.cuda()
        with torch.no_grad():
            time1 = datetime.datetime.now()
            outputs = net(images)

            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            output = non_max_suppression(output, 1, conf_thres=0.5)
            if ((datetime.datetime.now() - time1).seconds > 5):
                logging.info('Batch %d time is too long ' % (step))
                n_gt = 1
                break
            # print("time2", (datetime.datetime.now() - time1).seconds*1000+(datetime.datetime.now() - time1).microseconds//1000)
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
                        for x1, y1, x2, y2, conf, obj_conf, obj_pred in sample_pred[
                                    sample_pred[:, 6] == obj_cls]:
                            box_pred = torch.cat([coord.unsqueeze(0) for coord in [x1, y1, x2, y2]]).view(1, -1)
                            iou = bbox_iou(box_pred, box_gt)
                            if iou >= config["iou_thres"]:
                                correct += 1
                                break
        if n_gt:
            logging.info('Batch [%d/%d] mAP: %.5f' % (step, len(dataloader), float(correct / n_gt)))

    logging.info('Mean Average Precision: %.5f' % float(correct / n_gt))
def train(config):
    config["global_step"] = config.get("start_step", 0)
    is_training = False if config.get("export_onnx") else True


    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Optimizer and learning rate
    optimizer = _get_optimizer(config, net)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    # lr_scheduler = optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=config["lr"]["decay_step"],
    #     gamma=config["lr"]["decay_gamma"])

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)

    # Start the training loop
    logging.info("Start training.")
    dataload_len=len(dataloader)
    epoch_size = 4
    start = time.time()
    pruned_pct = 0
    global index, pruned_book, num_pruned
    global num_weights
    global weight_masks, bias_masks
    for epoch in range(config["epochs"]):
        if epoch%4==0:
            index = 0
            num_pruned = 0
            num_weights = 0
            net.apply(prune)
            torch.save(net.state_dict(), '%s/%.4f_%04d.weights' % (checkpoint_dir, 0.01, 1))
            print('previously pruned: %.3f %%' % (100 * (pruned_pct)))
            print('number pruned: %.3f %%' % (100 * (num_pruned / num_weights)))
            new_pruned = num_pruned / num_weights - pruned_pct
            pruned_pct = num_pruned / num_weights
            # if new_pruned <= 0.01:
            #     time_elapse = time.time() - start
            #     print('training time:', str(timedelta(seconds=time_elapse)))
            #     break
        recall = 0
        mini_step = 0
        for step, samples in enumerate(dataloader):
            index = 0
            images, labels = samples["image"], samples["label"]
            start_time = time.time()
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

            net.apply(set_grad)
            optimizer.step()
            _loss = loss.item()
            # example_per_second = config["batch_size"] / duration
            lr = optimizer.param_groups[0]['lr']

            strftime = datetime.datetime.now().strftime("%H:%M:%S")
            recall += losses[7] / 3
            print(
                '%s [Epoch %d/%d,batch %03d/%d loss:x %.5f,y %.5f,w %.5f,h %.5f,conf %.5f,cls %.5f,total %.5f,rec %.3f,avrec %.3f %.3f]' %
                (strftime, epoch, config["epochs"], step, dataload_len,
                 losses[1], losses[2], losses[3],
                 losses[4], losses[5], losses[6],
                 _loss, losses[7] / 3, recall / (step+1), lr))

        if (epoch % 2 == 0 and recall / len(dataloader) > 0.5) or recall / len(dataloader) > 0:
            # torch.save(net.state_dict(), '%s/%.4f_%04d.weights' % (checkpoint_dir, recall / len(dataloader), epoch))
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
    config["batch_size"] *= len(config["parallels"])

    # Create sub_working_dir
    sub_working_dir = '{}/{}/size{}x{}_try{}/{}'.format(
        config['working_dir'], config['model_params']['backbone_name'], 
        config['img_w'], config['img_h'], config['try'],
        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    train(config)

