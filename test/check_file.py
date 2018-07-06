import os
import shutil
path_xml = r'\\192.168.55.39\team-CV\dataset\tiny_data_0627\test\Annotations/'
path_jpg = r'\\192.168.55.39\team-CV\dataset\tiny_data_0627\test\JPEGImages/'
xmls = os.listdir(path_xml)
jpgs = os.listdir(path_jpg)

for img in jpgs:
    print(img[:-4])

print(len(xmls))
print(len(jpgs))
xml_list = [c.split('.')[0] for c in xmls]
jpg_list = [c.split('.')[0] for c in jpgs]
# for i in range(len(jpgs)):
#     if xmls[i].split('.')[0] != jpgs[i].split('.')[0]:
#         print(xmls[i] ,jpgs[i])
#         break
c_xml = [c for c in xml_list if c not in jpg_list]
c_jpg = [c for c in jpg_list if c not in xml_list]
print('c_xml',c_xml)
print('c_jpg',c_jpg)