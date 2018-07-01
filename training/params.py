TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "../weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": "13,18, 19,31, 23,55, 26,80, 37,67, 40,50, 45,36, 69,206, 81,122",
        "classes": 1,
    },
    "lr": {
        "backbone_lr": 0.001,
        "other_lr": 0.01,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 20,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "adam",
        "weight_decay": 4e-05,
    },
    "batch_size": 6,
    # "train_path": "../data/coco/trainvalno5k.txt",
    "train_path": r"D:\data\VOC2007",
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
