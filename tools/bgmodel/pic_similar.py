import datetime

from skimage.measure import compare_ssim
import cv2
import os

import voc
import numpy as np


# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return 1 - n / 64


def pHash(img):
    """get image pHash value"""
    # 加载并调整图片为32x32灰度图片
    # img=cv2.imread(imgfile, 0)


    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    # cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(32, 32)

    # 把二维list变成一维list
    img_list = vis1.flatten()

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])


base_img=r"\\192.168.55.38\Team-CV\cam2pick\bg_pic\bj_bd003\backGround\0719_035729_215521.jpg"
base_img=r"\\192.168.55.38\Team-CV\cam2pick\bg_pic\sh_wuding\backGround\0713_163023_569660.jpg"
base_img=r"\\192.168.55.38\Team-CV\cam2pick\bg_pic\bj_800\backGround\0716_225551_799320.jpg"
base_img=r"D:\data\test\rec_pic\0720_090352_840781.jpg"
bath_path = r"\\192.168.55.38\Team-CV\cam2pick\camera_pic_0718\bj_bd003"
bath_path = r"\\192.168.55.38\Team-CV\cam2pick\camera_pic_0717\bj_800"
bath_path = r"D:\data\test"
# bath_path = r"\\192.168.55.38\Team-CV\cam2pick\camera_pic_0719\bj_bd003\video\201807200000"
pic_path = bath_path + "/rec_pic/"

xml_path = bath_path + "/Annotations/"

result =bath_path + "/0720/"

result_mouse =bath_path + "/0720_mouse/"

os.makedirs(result, exist_ok=True)
os.makedirs(result_mouse, exist_ok=True)

xmls = os.listdir(xml_path)


model_pic=cv2.imread(base_img,0)


max_similar=1.1
index=0
for path in xmls:

    img=cv2.imread(pic_path+path[:-3]+"jpg",0)
    # if index==0:
    #     model_pic=img
    #     index=1

    if img is None:
        print("img is none", pic_path + path[:-3] + "jpg")
        continue
    boxs = voc.get_boxs(xml_path + path)
    if len(boxs)==0:
        print("boxs is none",pic_path+path[:-3]+"jpg")
        continue
    for box in boxs:
        pic_box=img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        base_box=model_pic[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]

        time1=datetime.datetime.now()
        ssim = compare_ssim(pic_box, base_box)#,multichannel=True)  # calculate their ssim
        print("compare_ssim" ,(datetime.datetime.now()-time1).microseconds)


        if ssim>max_similar:
            max_similar=ssim
        if (ssim > 0.8):  # if they are similar enough, delete one of them
            cv2.imwrite(result + path[:-3] + "jpg",img)
            print("ssim img" ,ssim,pic_path+path[:-3]+"jpg")
        else:
            cv2.imwrite(result_mouse  + path[:-3] + "jpg", img)
            print("    no ssim",ssim, pic_path + path[:-3] + "jpg")
        time1 = datetime.datetime.now()
        hash1 = pHash(pic_box)
        hash2 = pHash(base_box)
        n = cmpHash(hash1, hash2)
        print('pHash度：', n, "--time=", (datetime.datetime.now() - time1).microseconds)
print("max_similar",max_similar)
