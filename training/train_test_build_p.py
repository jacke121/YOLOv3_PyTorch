import os
import random

# xmlfilepath = r"F:\data_set\frcnn\bj_yuanyang\VOC2007\Annotations"
saveBasePath = r"\\192.168.55.39\team-CV\dataset\origin_all_datas\all_scenes"

trainval_percent = 0.5
train_percent = 0.5
# total_xml = os.listdir(saveBasePath )

# scene_paths = [path for path in os.listdir(saveBasePath) if path != 'ImageSets']
scene_paths=["bj_yy01","bj_yy02","bj_yy03","bj_yy04","bj_yyh","sh_wd"]
total_xml = []
for scene_path in scene_paths:
    anno_path = os.path.join(saveBasePath,scene_path,'Annotations')
    files = os.listdir(anno_path)
    for file in files:
        total_xml.append('/%s/Annotations/%s'%(scene_path,file))

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("traub suze", tr)
ftrainval = open(os.path.join(saveBasePath, 'ImageSets/Main/trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'ImageSets/Main/test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'ImageSets/Main/train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'ImageSets/Main/val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()