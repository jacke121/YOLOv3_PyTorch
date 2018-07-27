import os
import shutil
path_xml = r'\\192.168.55.39\team-CV\dataset\origin_all_datas\_2train\bj_yy01\Annotations/'
path_jpg = r'\\192.168.55.39\team-CV\dataset\origin_all_datas\_2train\bj_yy01\JPEGImages/'
path_jpg = r'\\192.168.55.38\Team-CV\cam2pick\camera_pic_0718\bj_800\mouse_null_all\rec_pic/'
# path_xml = r'E:\Team-CV\tmp\lbg\Annotations/'
# path_jpg = r'E:\Team-CV\tmp\lbg\rec_pic/'
xmls = os.listdir(path_xml)
jpgs = os.listdir(path_jpg)

for img in jpgs:
    print(img[:-4])

# print(len(xmls))
# print(len(jpgs))
# xml_list = [c.split('.')[0] for c in xmls]
# jpg_list = [c.split('.')[0] for c in jpgs]
# # for i in range(len(jpgs)):
# #     if xmls[i].split('.')[0] != jpgs[i].split('.')[0]:
# #         print(xmls[i] ,jpgs[i])
# #         break
# c_xml = [c for c in xml_list if c not in jpg_list]
# c_jpg = [c for c in jpg_list if c not in xml_list]
#
# for file in c_xml:
#     pass
#     # os.remove(path_xml+file+".xml")
#
# print('c_xml',c_xml)
# print('c_jpg',c_jpg)