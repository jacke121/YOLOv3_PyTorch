from __future__ import division

import cv2

from models import *
from utils.preprocess import letterbox_image
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
import os
import sys
import time
import datetime
import argparse

import torch
from torch.autograd import Variable

from PIL import Image,ImageDraw,ImageFont
from numpy import unicode
parser = argparse.ArgumentParser()
parser.add_argument('--model_config_path', type=str, default='config/yolov3_2cls.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='checkpoints/41.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
opt = parser.parse_args()
print(opt)

os.makedirs('output', exist_ok=True)

cuda = torch.cuda.is_available()
module_defs=parse_model_config(opt.model_config_path)
hyperparams     = module_defs[0]
hyperparams["is_training"]=1
anchors=hyperparams["anchors"]
anchors = [int(x) for x in anchors.split(",")]
anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
module_defs[83]["anchors"]=anchors
module_defs[95]["anchors"]=anchors
module_defs[107]["anchors"]=anchors
hyperparams['height']=hyperparams['width']=opt.img_size
# Set up model
model = Darknet(module_defs, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode
ttfont = ImageFont.truetype("msyh.ttf", 10)
classes = ["mouse"]
labels = ["人体", "自行车", "汽车", "摩托车", "飞机", "公共汽车", "火车", "卡车", \
          "船", "交通灯", "消防栓", "停止标识", "停车收费表", "长凳", \
          "鸟", "猫", "狗", "马", "羊", "牛", "象", "熊", "斑马", "长颈鹿", \
          "背包", "伞", "手提包", "领带", "行李箱", "飞盘", "滑雪板", "滑雪板", \
          "运动球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", \
          "网球拍", "瓶子", "酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", \
          "苹果", "三明治", "橙子", "花椰菜", "胡萝卜", "热狗", "比萨饼", "甜甜圈", "蛋糕", \
          "椅子", "沙发", "盆栽", "床", "餐桌", "马桶", "显示器", "笔记本电脑", "鼠标", \
          "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机", "水槽", "冰箱", \
          "书", "钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"]
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None,label=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = label
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score
def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    #unsqueeze 扩展维度
    return img_, orig_im, dim
def prep_images(imgs, inp_dim):
    img_=[]
    dim=1,1
    for image in imgs:
        orig_im = image
        dim = orig_im.shape[1], orig_im.shape[0]
        img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
        img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_.append(img)
    #unsqueeze 扩展维度
    img_=np.asarray(img_)
    img_=torch.from_numpy(img_).float().div(255.0)
    return img_,dim

def draw_boxes(image, boxes, labels, obj_thresh):
    for box in boxes:
        cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 1)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        # draw.text((box.xmin, box.ymin - 13), unicode(labels[box.classes] + " " + str(box.objness)), fill=(255, 0, 0),
        #           font=ttfont)
        draw.text((box.xmin, box.ymin - 13), unicode(labels[box.classes]), fill=(255, 0, 0), font=ttfont)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


    return image

def detect_imgs(frames):
    imgs, dim = prep_images(frames, opt.img_size)
    if cuda:
        imgs = imgs.cuda()

    input_imgs = Variable(imgs)
    time_old = datetime.datetime.now()
    with torch.no_grad():
        detections = model(input_imgs)
        print('cost time', (datetime.datetime.now() - time_old).microseconds)
        detections = non_max_suppression(detections, 1, opt.conf_thres, opt.nms_thres)
    # The amount of padding that was added
    pad_x = max(frames[0].shape[0] - frames[0].shape[1], 0) * (opt.img_size / max(frames[0].shape))
    pad_y = max(frames[0].shape[1] - frames[0].shape[0], 0) * (opt.img_size / max(frames[0].shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x
    # Draw bounding boxes and labels of detections
    if detections is not None:
        # unique_labels = detections[0][:, -1].cpu().unique()
        # n_cls_preds = len(unique_labels)

        index = 0
        for detection in detections:
            if detection is None:
                index += 1
                continue
            boxes = []
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                # print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                box_h = ((y2 - y1) / unpad_h) * frames[index].shape[0]
                box_w = ((x2 - x1) / unpad_w) * frames[index].shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * frames[index].shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * frames[index].shape[1]
                box = BoundBox(x1, y1, x1 + box_w, y1 + box_h, cls_conf.item(), int(cls_pred), classes[int(cls_pred)])
                boxes.append(box)
            img_show = draw_boxes(frames[index], boxes, labels, opt.conf_thres)
            img_show = cv2.resize(img_show, (img_show.shape[1], img_show.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
            # outVideo.write(img_show)
            cv2.imshow("ai", img_show)
            cv2.waitKey()
            index += 1
def batch_file():
    path = r'D:\data\bj_yuanyang01\JPEGImages/'
    files = os.listdir(path)
    index=0
    frames=[]
    for file in files:
        # while cap.isOpened():
        frame = cv2.imread(path + file)
        frames.append(frame)
        index+=1
        if index%4==0:
            detect_imgs(frames)
            frames.clear()

def open_file():
    # cap = cv2.VideoCapture("./007.avi")
    index=0
    #load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    img_i = 0
    start = time.time()
    path = r'D:\data\bj_yuanyang01\JPEGImages/'
    files = os.listdir(path)
    for file in files:
    # while cap.isOpened():
        frame = cv2.imread(path+file)
        # ret, frame = cap.read()
        img_i+=1
        img, orig_im, dim = prep_image(frame, opt.img_size)

        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if cuda:
            im_dim = im_dim.cuda()
            img = img.cuda()
        input_imgs = Variable(img)
        time_old = datetime.datetime.now()
        with torch.no_grad():
            detections = model(input_imgs)
            print('cost time',img_i,(datetime.datetime.now()-time_old).microseconds)
            detections = non_max_suppression(detections, 1, opt.conf_thres, opt.nms_thres)

        # The amount of padding that was added
        pad_x = max(frame.shape[0] - frame.shape[1], 0) * (opt.img_size / max(frame.shape))
        pad_y = max(frame.shape[1] - frame.shape[0], 0) * (opt.img_size / max(frame.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x
        # Draw bounding boxes and labels of detections
        if detections is not None and detections[0] is not None:
            print(detections[0])
            unique_labels = detections[0][:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            boxes=[]
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
                # print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * frame.shape[0]
                box_w = ((x2 - x1) / unpad_w) * frame.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * frame.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * frame.shape[1]


                box = BoundBox(x1, y1, x1+box_w, y1 + box_h, cls_conf.item(),int(cls_pred), classes[int(cls_pred)])
                boxes.append(box)
                # bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                #                          edgecolor=color,
                #                          facecolor='none')
                # Add the bbox to the plot
                # ax.add_patch(bbox)
                # plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                #          bbox={'color': color, 'pad': 0})
            img_show = draw_boxes(frame, boxes, labels, opt.conf_thres)
            img_show = cv2.resize(img_show, (img_show.shape[1], img_show.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
            # outVideo.write(img_show)
            cv2.imshow("ai", img_show)
            cv2.waitKey()
if __name__ == '__main__':
    # open_file()
    batch_file()
