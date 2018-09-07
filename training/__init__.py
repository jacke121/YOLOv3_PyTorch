# import os
import datetime

from sklearn.neighbors import KDTree
import numpy as np
# path=r"D:\deeplearning\yolov3\YOLOv3_PyTorch\training\checkpoints\result"
#
# xmls = os.listdir(path)
#
# for img in xmls:
#     print(img[:-4])

#
# #测试 BallTree
# from sklearn.neighbors import BallTree
# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# bt = BallTree(X,leaf_size=30,metric="euclidean")
# print(bt)
# x=[1,2]
# print(bt.query(X, k=2, return_distance=False))

from sklearn.neighbors import KDTree,BallTree
import numpy as np # 快速操作结构数组的工具


print(35345)