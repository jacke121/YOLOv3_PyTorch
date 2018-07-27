import json
import threading

import datetime
from numba import jit
import numpy as np
import os
import cv2

import time
import io
import requests
from PIL import Image
# url = 'http://192.168.25.56:8000/pic_class/'
# url_path = 'http://192.168.25.89:8000/detect_bypath/'
@jit
def initial_background(I_gray, N):#初始化
    I_pad = np.pad(I_gray, 1, 'symmetric')
    height = I_pad.shape[0]
    width = I_pad.shape[1]
    samples = np.zeros((height, width, N))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            for n in range(N):
                x, y = 0, 0
                while (x == 0 and y == 0):
                    x = np.random.randint(-1, 1)
                    y = np.random.randint(-1, 1)
                ri = i + x
                rj = j + y
                samples[i, j, n] = I_pad[ri, rj]
    samples = samples[1:height - 1, 1:width - 1]
    return samples

@jit
def vibe_detection(I_frame,I_gray, samples, _min, N, R):#建模预测
    height = I_gray.shape[0]
    width = I_gray.shape[1]
    segMap = np.zeros((height, width)).astype(np.uint8)
    for i in range(height):
        for j in range(width):
            count, index, dist = 0, 0, 0
            while count < _min and index < N:
                dist = np.abs(I_gray[i, j] - samples[i, j, index])
                if dist < R:
                    count += 1
                index += 1
            if count >= _min:
                r = np.random.randint(0, N - 1)
                if r == 0:
                    r = np.random.randint(0, N - 1)
                    samples[i, j, r] = I_gray[i, j]
                r = np.random.randint(0, N - 1)
                if r == 0:
                    x, y = 0, 0
                    while (x == 0 and y == 0):
                        x = np.random.randint(-1, 1)
                        y = np.random.randint(-1, 1)
                    r = np.random.randint(0, N - 1)
                    ri = i + x
                    rj = j + y
                    samples[ri, rj, r] = I_gray[i, j]
            else:
                segMap[i, j] = 255
    segMap = cv2.erode(segMap, None, iterations=3)
    segMap = cv2.dilate(segMap, None, iterations=3)
    segMap = cv2.dilate(segMap, None, iterations=3)
    segMap = cv2.erode(segMap, None, iterations=3)
    ret, binary = cv2.threshold(segMap, 127, 255, cv2.THRESH_BINARY)
    hierarchy, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return segMap, samples,contours
def savemat(image,box):#保存矩阵
    # time_old = datetime.datetime.now()
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = image[y1:y1 + hight, x1:x1 + width]
    # print('savemat',(datetime.datetime.now()-time_old).microseconds)
    return cropImg,hight,width
i = 0

def load_image(single_mat):
    # image_data = Image.open(image_file)
    img = Image.fromarray(single_mat)
    output = io.BytesIO()
    img.save(output, format='JPEG') # format=image_data.format
    # print(img.format)  # 输出的不一定是JPEG也可能是PNG
    # image_data.close()
    data_bin = output.getvalue()
    output.close()
    return data_bin

def saveImg(rectPath,img_list,timeStr):
    for i,img in enumerate(img_list):
        cv2.imwrite(rectPath + '/' + timeStr + 'id_' + str(i) + '.jpg', img)

def save_ime(image227,img_list,boundlist):
    for i,img in enumerate(img_list):
        timestr = datetime.datetime.now().strftime('%m%d_%H%M%S_%d')
        cv2.imwrite(image227+ '/' + timestr +'_%04d_%5f.jpg'%(i,boundlist[i][5]), img)
def save_origin(image227,img_list):
    for i,img in enumerate(img_list):
        timestr = datetime.datetime.now().strftime('%m%d_%H%M%S_%d')
        cv2.imwrite(image227+ '/' + timestr +'_%04d.jpg'%(i), img)
def save2H(icounter,data,mat_list,neg_list,origin_list,deviceid):
    dir_countername = icounter // 10000
    outdata_root = r'F:\mmy\data_set1\django_data%d/' % (dir_countername) + str(deviceid)
    pic_origin = outdata_root + '/pic_origin/'
    pic_posi = outdata_root + '/pic_posi/'
    pic_neg = outdata_root + '/pic_neg/'
    if not os.path.isdir(pic_origin):
        os.makedirs(pic_origin)
    if not os.path.isdir(pic_posi):
        os.makedirs(pic_posi)
    if not os.path.isdir(pic_neg):
        os.makedirs(pic_neg)
    save_ime(pic_posi, mat_list,data["bbox_mouse"])
    save_ime(pic_neg, neg_list,data["bbox_neg"])
    save_origin(pic_origin, origin_list)
@jit
def c2mat(frame,c):
    boundbox = []
    mat_list = []
    for m in c:
        rect = cv2.minAreaRect(m)
        x, y, w, h = cv2.boundingRect(m)
        if w < 5 or h < 5 or w * h < 40 or w > 35 or h > 35 or w * h > 800 or w / h > 7 or w / h < 1 / 8:
            continue
        boundbox.append((x, y, w, h))
        box = np.int0(cv2.boxPoints(rect))
        cropImg, height, width = savemat(frame, box)
        if cropImg.shape[0] > 0 and cropImg.shape[1] > 0:
            cropImg = cv2.resize(cropImg, (227, 227), interpolation=cv2.INTER_CUBIC)
            # cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 255, 0), 2)
            mat_list.append(cropImg)
        else:
            pass
    return boundbox,mat_list
@jit
def mat_output(frame,update_flag = True,samples = None):
    mat_list = []
    boundbox=[]
    N = 20
    R = 20
    _min = 2
    time_old = datetime.datetime.now()
    if not update_flag:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        samples = initial_background(frame, N)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        segMap, samples, contours = vibe_detection(frame, gray, samples, _min, N, R)  # 求取
        print('vibe_detection', (datetime.datetime.now() - time_old).microseconds)
        if len(contours) > 0:
            c = sorted(contours, key=cv2.contourArea, reverse=True)
            boundbox, mat_list = c2mat(frame,c)#0.25秒
            print('c2mat', (datetime.datetime.now() - time_old).microseconds)
                # cv2.drawContours(frame, [box], -1, (0, 0, 255), 2)
    return samples,mat_list,boundbox


# @jit
def face_mosaic(frame,face_bbox):
    for (x, y, w, h) in face_bbox:
        # cv2.rectangle(frame,(x, y), (x + w, y + h), (0, 0, 255),2)
        cutImg = frame[max(0,y-20):min(y + h+20,frame.shape[0]),max(0, x-20):min(x + w+20,frame.shape[1])]
        cutFace = cutImg.shape[:2][::-1]
        cutImg = cv2.resize(cutImg, (int(cutFace[0] / 30), int(cutFace[1] / 30)))
        cutImg = cv2.resize(cutImg, cutFace, interpolation=cv2.INTER_NEAREST)
        frame[max(0,y-20):min(y + h+20,frame.shape[0]),max(0, x-20):min(x + w+20,frame.shape[1])] = cutImg
    return frame


if __name__ == '__main__':
    output="c:/output/bj_test02"
    os.makedirs(output, exist_ok=True)
    rootDir = r'\\192.168.55.38\Team-CV\cam2pick\camera_pic_0717\bj_test02\rec_pic'
    image_file = os.path.join(rootDir, os.listdir(rootDir)[0])

    frame = cv2.imread(image_file)
    samples,_,_ = mat_output(frame,update_flag = False)

    # url = 'http://127.0.0.1:8000/pic_class/'
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        frame = cv2.imread(path)
        if frame is None:
            continue
        samples, mat_list,boundbox = mat_output(frame,samples=samples)
        if len(mat_list)>0:
            cv2.imwrite(output+"/" + os.path.basename(path), frame)
        # cv2.imshow("img", frame)
        # cv2.waitKey(1)

        # print(code,'---',ret_data)
        # frame = cv2.imread(path)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # segMap, samples, contours = vibe_detection(frame,gray, samples, _min, N, R)#求取
        # if len(contours)>0:
        #     c = sorted(contours, key=cv2.contourArea, reverse=True)
        #     for m in c:
        #         rect = cv2.minAreaRect(m)
        #         x, y, w, h = cv2.boundingRect(m)
        #         if w<6 or h<6 or w*h<40:
        #             continue
        #         cv2.rectangle(frame, (x-3, y-3), (x + w+3, y + h+3), (0, 255, 0), 2)
        #         box = np.int0(cv2.boxPoints(rect))
        #         cropImg, height, width = savemat(frame, box)
        #         cv2.imwrite('image222/' + str(i) + '.jpg', cropImg)
        #         # cv2.drawContours(frame, [box], -1, (0, 0, 255), 2)
        #     cv2.imshow("img", frame)
        #     i+=1
        # if cv2.waitKey(20) and 0xff == ord('q'):
        #     break
    cv2.destroyAllWindows()