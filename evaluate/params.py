TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": "14,19, 15,33, 24,75, 28,55, 33,94, 38,34, 47,71, 49,46, 91,144",
        "classes": 1,
    },
    "batch_size": 16,
    "iou_thres": 0.5,
    "val_path": r"D:\data\tiny_data\VOC2007",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    # "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
    "pretrain_snapshot": "../training/checkpoints/18.weights",
    # "pretrain_snapshot": r"C:\Users\sbdya\Desktop\tmp/140.weights",
}
