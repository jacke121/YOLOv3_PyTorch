import datetime
import os

import cv2
import shutil

import voc
import numpy as np

bath_path = r"\\192.168.55.38\Team-CV\cam2pick\camera_pic_0717\bj_test02"
pic_path = bath_path + "/rec_pic/"

xml_path = bath_path + "/Annotations/"

import numpy as np

xmls = os.listdir(xml_path)


samples=[]
repeat_count=[]
for path in xmls:
    boxs = voc.get_boxs(xml_path + path)

    time1=datetime.datetime.now()
    data = np.array(boxs)
    idex = np.lexsort([data[:, 3], data[:, 2], data[:, 1], data[:, 0]])
    sorted_data = data[idex, :]

    sorted_data=sorted_data.reshape(len(sorted_data) * 4)
    min_distence = 9999
    min_index=0
    if len(samples)>0:
        for inxdex, sample in enumerate(samples):
            a = sample
            b = sorted_data
            if len(sample) > len(sorted_data):
                b = np.zeros(len(sample), dtype=np.int)
                b[0:len(sorted_data)] = sorted_data
            elif len(sample) < len(sorted_data):
                a = np.zeros(len(sorted_data), dtype=np.int)
                a[0:len(sample)] = sample
            distance = np.linalg.norm(a - b)
            if (min_distence>distance):
                min_distence=distance
                min_index=inxdex
        if min_distence>3 and min_distence<9999:
            pass
            # print(pic_path + path[:-4] + ".jpg", min_distence)
            # shutil.copyfile(xml_path + path, "output/Annotations/"+path)
            # shutil.copyfile(pic_path + path[:-4] + ".jpg", "output/rec_pic/" + path[:-4] + ".jpg")
        elif repeat_count[min_index]<200:
            pass
            # print(pic_path + path[:-4] + ".jpg", repeat_count[min_index])
            # shutil.copyfile(xml_path + path, "output/Annotations/" + path)
            # shutil.copyfile(pic_path + path[:-4] + ".jpg", "output/rec_pic/" + path[:-4] + ".jpg")
        else:
            repeat_count[min_index]+=1

    if len(samples) > 5:
        min_index = repeat_count.index(min(repeat_count))
        samples[min_index] = sorted_data
        pass
    else:
        samples.append(sorted_data)
        repeat_count.append(0)
    print("time1",(datetime.datetime.now()-time1).microseconds)

class BoxRepeat():
    repeat_count = []

    samples = []
    def box_repeat(self,frame, bbox_mouse,timestr):
        time1 = datetime.datetime.now()
        if len(bbox_mouse)==0:
            return
        data = np.array(bbox_mouse)
        idex = np.lexsort([data[:, 3], data[:, 2], data[:, 1], data[:, 0]])
        sorted_data = data[idex, :]

        sorted_data = sorted_data.reshape(len(sorted_data) * 4)
        min_distence = 9999
        min_index = 0

        if len(self.samples) == 0:
            self.samples.append(sorted_data)
            self.repeat_count.append(0)
        else:
           for inxdex, sample in enumerate(self.samples):
               a = sample
               b = sorted_data
               if len(sample) > len(sorted_data):
                   b = np.zeros(len(sample), dtype=np.int)
                   b[0:len(sorted_data)] = sorted_data
               elif len(sample) < len(sorted_data):
                   a = np.zeros(len(sorted_data), dtype=np.int)
                   a[0:len(sample)] = sample
               distance = np.linalg.norm(a - b)
               if (min_distence > distance):
                   min_distence = distance
                   min_index = inxdex
           if min_distence > 3 and min_distence < 9999:
               if len(self.samples) > 5:
                   if max(self.repeat_count) > 50 and min(self.repeat_count) < 5:
                       min_index = self.repeat_count.index(min(self.repeat_count))
                       self.samples[min_index] = sorted_data
                       self.repeat_count[min_index] = 0
               else :
                   self.samples.append(sorted_data)
                   self.repeat_count.append(0)
               cv2.imwrite(self.rec_path + '../repeat_no/' + timestr + '.jpg', frame)
               # print(pic_path + path[:-4] + ".jpg", min_distence)
               # shutil.copyfile(xml_path + path, "output/Annotations/"+path)
               # shutil.copyfile(pic_path + path[:-4] + ".jpg", "output/rec_pic/" + path[:-4] + ".jpg")
           elif self.repeat_count[min_index] < 200:
               cv2.imwrite(self.rec_path + '../repeat2/' + timestr + '.jpg', frame)
               self.repeat_count[min_index] += 1
               print("repeat_count",min_index, self.repeat_count[min_index])
               # shutil.copyfile(xml_path + path, "output/Annotations/" + path)
               # shutil.copyfile(pic_path + path[:-4] + ".jpg", "output/rec_pic/" + path[:-4] + ".jpg")
           else:
               cv2.imwrite(self.rec_path + '../repeat200/' + timestr + '.jpg', frame)
               self.repeat_count[min_index] += 1
        print("box_repeat time1", (datetime.datetime.now() - time1).microseconds)
if __name__ == '__main__':
    box_re= BoxRepeat()

    xmls = os.listdir(xml_path)

    for path in xmls:
        boxs = voc.get_boxs(xml_path + path)
        data = np.array(boxs)

        box_re.box_repeat()


        time1 = datetime.datetime.now()




