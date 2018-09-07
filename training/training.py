# coding='utf-8'
import os
import sys
import time
import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim

torch.backends.cudnn.benchmark = True

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from common.coco_dataset import COCODataset


TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained":"", #  set empty to disable
        # "backbone_pretrained":"../weights/mobilenetv2_weights.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": "16,24, 23,39, 25,84, 31,66, 42,54, 46,38, 56,81, 59,121, 74,236",
        "classes": 1,
    },
    "lr": {
        "backbone_lr": 0.05,
        "other_lr": 0.05,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.2,
        "decay_step": 15,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "batch_size": 60,
    # "train_path": "../data/coco/trainvalno5k.txt",
    "train_path": r"\\192.168.55.73\team-CV\dataset\origin_all_datas\_2train",
    "epochs": 2001,
    "img_h": 352,
    "img_w": 352,
    # "parallels": [0,1,2,3],                         #  config GPU device
    "parallels": [0],                         #  config GPU device
    "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir
    "pretrain_snapshot": "",
    # "pretrain_snapshot": r"F:\Team-CV\checkpoints\shuffle_v2\0.8204_0167.weights",                        #  load checkpoint
    # "pretrain_snapshot": r"F:\Team-CV\checkpoints\torch_yolov0802/0.6185_0070.weights",
    "evaluate_type": r"F:\Team-CV\checkpoints\torch_yolov0824/0.9654_1667.weights",
    "try": 0,
    "export_onnx": False,
}
checkpoint_dir=r"F:\Team-CV\checkpoints\torch_yolov0824_"
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

    # Only export onnx
    # if config.get("export_onnx"):
        # real_model = net.module
        # real_model.eval()
        # dummy_input = torch.randn(8, 3, config["img_h"], config["img_w"]).cuda()
        # save_path = os.path.join(config["sub_working_dir"], "pytorch.onnx")
        # logging.info("Exporting onnx to {}".format(save_path))
        # torch.onnx.export(real_model, dummy_input, save_path, verbose=False)
        # logging.info("Done. Exiting now.")
        # sys.exit()

    # Evaluate interface
    # if config["evaluate_type"]:
        # logging.info("Using {} to evaluate model.".format(config["evaluate_type"]))
        # evaluate_func = importlib.import_module(config["evaluate_type"]).run_eval
        # config["online_net"] = net

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLayer(config["batch_size"],i,config["yolo"]["anchors"][i],
                                     config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # DataLoader
    dataloader = torch.utils.data.DataLoader(COCODataset(config["train_path"],
                                                         (config["img_w"], config["img_h"]),
                                                         is_training=True,is_scene=True),
                                             batch_size=config["batch_size"],
                                             shuffle=True,drop_last=True, num_workers=0, pin_memory=True)

    # Start the training loop
    logging.info("Start training.")
    dataload_len=len(dataloader)
    best_acc=0.5
    for epoch in range(config["epochs"]):

        recall = 0
        mini_step = 0
        for step, samples in enumerate(dataloader):
            images, labels = samples["image"], samples["label"]
            config["global_step"] += 1
            # Forward and backward
            optimizer.zero_grad()
            outputs = net(images.cuda())
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
            lr = optimizer.param_groups[0]['lr']

            strftime = datetime.datetime.now().strftime("%H:%M:%S")
            # if (losses[7] / 3 >= recall / (step + 1)):#mini_batch为0走这里
            recall += losses[7] / 3
            print(
                '%s [Epoch %d/%d,batch %03d/%d loss:x %.5f,y %.5f,w %.5f,h %.5f,conf %.5f,cls %.5f,total %.5f,rec %.3f,avrec %.3f %.3f]' %
                (strftime, epoch, config["epochs"], step, dataload_len,
                 losses[1], losses[2], losses[3],
                 losses[4], losses[5], losses[6],
                 _loss, losses[7] / 3, recall / (step + 1), lr))

        if recall / len(dataloader) > best_acc:
            best_acc=recall / len(dataloader)
            if epoch>0:
                torch.save(net.state_dict(), '%s/%.4f_%04d.weights' % (checkpoint_dir, recall / len(dataloader), epoch))

        lr_scheduler.step()
        net.train(is_training)
        torch.cuda.empty_cache()
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
    logging.info("Using " + config["optimizer"]["type"] + " optimizer.")
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
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    config = TRAINING_PARAMS
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

if __name__ == "__main__":
    main()
