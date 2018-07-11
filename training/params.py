TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        # "backbone_pretrained":"", #  set empty to disable
        "backbone_pretrained":"../weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": "13,19, 19,33, 24,57, 24,83, 36,69, 43,51, 43,36, 67,106, 96,197",
        "classes": 1,
    },
    "lr": {
        "backbone_lr": 0.002,
        "other_lr": 0.02,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.4,
        "decay_step": 10,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "adam",
        "weight_decay": 4e-05,
    },
    "batch_size": 10,
    # "train_path": "../data/coco/trainvalno5k.txt",
    "train_path": r"\\192.168.55.39\team-CV\dataset\original_0706",
    "epochs": 2001,
    "img_h": 416,
    "img_w": 416,
    # "parallels": [0,1,2,3],                         #  config GPU device
    "parallels": [0],                         #  config GPU device
    "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir
    "pretrain_snapshot": "",                        #  load checkpoint
    "evaluate_type": "",
    "try": 0,
    "export_onnx": False,
}
