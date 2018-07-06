TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "", #  set empty to disable
        # "backbone_pretrained": "../weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": "13,18, 14,31, 25,52, 27,77, 27,29, 39,61, 44,40, 70,206, 73,119",
        "classes": 1,
    },
    "lr": {
        "backbone_lr": 0.001,
        "other_lr": 0.01,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.2,
        "decay_step": 15,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "adam",
        "weight_decay": 4e-05,
    },
    "batch_size": 10,
    # "train_path": "../data/coco/trainvalno5k.txt",
    "train_path": r"D:\data\Original",
    "epochs": 2001,
    "img_h": 416,
    "img_w": 416,
    # "parallels": [0,1,2,3],                         #  config GPU device
    "parallels": [0],                         #  config GPU device
    "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir
    "pretrain_snapshot": "checkpoints/142.weights",                        #  load checkpoint
    "evaluate_type": "", 
    "try": 0,
    "export_onnx": False,
}

TESTING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": "13,18, 14,31, 25,52, 27,77, 27,29, 39,61, 44,40, 70,206, 73,119",
        "classes": 1,
    },
    "batch_size": 1,
    "confidence_threshold": 0.8,
    "classes_names_path": "../data/coco2cls.names",
    "iou_thres": 0.2,
    "val_path": r"D:\data\VOC2007",
    "images_path":  r"D:\data\Original\JPEGImages/",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    "pretrain_snapshot": "",
    # "pretrain_snapshot": "../training/checkpoints/140.weights",
    "test_weights": r"E:\github\YOLOv3_PyTorch\training\checkpoints",
}