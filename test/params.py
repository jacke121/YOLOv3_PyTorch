TRAINING_PARAMS_2 = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "batch_size": 16,
    "confidence_threshold": 0.5,
    "images_path": "./images/",
    "classes_names_path": "../data/coco.names",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    "pretrain_snapshot": "../weights/darknet53_weights_pytorch.pth",
}

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
    "confidence_threshold": 0.5,
    "classes_names_path": "../data/coco2cls.names",
    "iou_thres": 0.5,
    "val_path": r"D:\data\VOC2007",
    "images_path":  r"D:\data\VOC2007\JPEGImages/",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    # "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
    "pretrain_snapshot": "../training/checkpoints/142.weights",
}