


import numpy as np
import cv2

import os
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)

from matplotlib import pyplot as plt

img = cv2.imread('2.jpg')
height, width = img.shape[:2]
mask = np.zeros(img.shape[:2],np.uint8)
# 背景模型
bgdModel = np.zeros((1,65),np.float64)
# 前景模型
fgdModel = np.zeros((1,65),np.float64)

rect = (10, 10, width - 30, height - 30)
# 使用grabCut算法
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

cv2.imshow('image', img)
k = cv2.waitKey(0)
#下面这种也能显示
plt.imshow(img),plt.colorbar(),plt.show()