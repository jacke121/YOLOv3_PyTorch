
import os
import numpy as np
path=r"D:\deeplearning\yolov3\YOLOv3_PyTorch\training\checkpoints\result"

xmls = os.listdir(path)

aaaa=[]
for img in xmls:
    bb=img[:-4].split("_")
    bb=np.float32(bb)
    aaaa.append(bb)

aaaa.sort(key=lambda x:x[1],reverse=True)

for i in range(len(aaaa)):
    print(aaaa[i])

    #[74.       0.90784]
    #[244.        0.90741]
    #[660.        0.90741]
