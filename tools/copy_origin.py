import os

import shutil

picpath=r"E:\github\YOLOv3_PyTorch\test\output_error"

source = r"\\192.168.55.38\Team-CV\cam2pick\camera_pic\sh_wuding\rec_pic_0706/"
imgs=os.listdir(picpath)

for img in imgs:
    shutil.copyfile(source+"/"+img, r"E:\github\YOLOv3_PyTorch\test\pic/"+img)

