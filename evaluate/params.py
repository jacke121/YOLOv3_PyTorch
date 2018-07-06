TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": "13,18, 14,31, 25,52, 27,77, 27,29, 39,61, 44,40, 70,206, 73,119",
        "classes": 1,
    },
    "batch_size": 16,
    "iou_thres": 0.5,
    "train_path": r"D:\data\Original",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    # "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
    "pretrain_snapshot": "../training/checkpoints/18.weights",
    # "pretrain_snapshot": r"C:\Users\sbdya\Desktop\tmp/140.weights",
}
