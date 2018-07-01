TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": "13,18, 19,31, 23,55, 26,80, 37,67, 40,50, 45,36, 69,206, 81,122",
        "classes": 1,
    },
    "batch_size": 16,
    "iou_thres": 0.5,
    "val_path": r"D:\data\VOC2007",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    # "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
    "pretrain_snapshot": "../training/checkpoints/96.weights",
}
