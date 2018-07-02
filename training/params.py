TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "", #  set empty to disable
        # "backbone_pretrained": "../weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": "14,19, 15,33, 24,75, 28,55, 33,94, 38,34, 47,71, 49,46, 91,144",
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
    "batch_size": 10,
    # "train_path": "../data/coco/trainvalno5k.txt",
    "train_path": r"D:\data\tiny_data\VOC2007",
    "epochs": 2001,
    "img_h": 416,
    "img_w": 416,
    # "parallels": [0,1,2,3],                         #  config GPU device
    "parallels": [0],                         #  config GPU device
    "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir
    "pretrain_snapshot": "checkpoints/100.pth",                        #  load checkpoint
    "evaluate_type": "", 
    "try": 0,
    "export_onnx": False,
}
