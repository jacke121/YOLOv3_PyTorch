TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": "13,19, 19,33, 24,57, 24,83, 36,69, 43,51, 43,36, 67,106, 96,197",
        "classes": 1,
    },
    "batch_size": 16,
    "iou_thres": 0.5,
    "train_path": r"\\192.168.55.39\team-CV\dataset\original_0706",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    # "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
    "pretrain_snapshot": "../training/checkpoints/0.9158_0020.weights",
    # "pretrain_snapshot": r"C:\Users\sbdya\Desktop\tmp/140.weights",
}
