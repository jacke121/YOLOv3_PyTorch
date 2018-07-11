import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes = ["mouse"]
def parse_voc_annotation(ann_dir, img_dir, cache_name, labels=[],org_path=None):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}

        trainval_path = os.path.join(org_path,'ImageSets/Main/trainval.txt')
        test_path = os.path.join(org_path,'ImageSets/Main/test.txt')
        with open(trainval_path, 'r') as file:
            train_files_ = file.readlines()
        with open(test_path, 'r') as file:
            test_files_ = file.readlines()
        all_files = train_files_+test_files_

        xml_files = [org_path + '%s.xml' % xml_file.strip('\n') for xml_file in all_files]
        jpg_files = [org_path + '%s.jpg' % jpg_file.strip('\n').replace('Annotations', 'JPEGImages') for jpg_file in all_files]
        print(len(xml_files))
        for ann in sorted(xml_files):
            img = {'object':[]}
            print(ann)

            tree = ET.parse(ann)
            # tree = ET.parse(in_file)
            root = tree.getroot()

            # img['filename'] = img_dir + root.find('filename').text
            img['filename'] = ann.replace('Annotations', 'JPEGImages').replace('xml','jpg')
            size = root.find('size')

            img['width'] = int(size.find('width').text)
            img['height'] =  int(size.find('height').text)
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for objnode in root.iter('object'):
                objs = {}
                difficult = 0  #objnode.find('difficult').text
                cls = objnode.find('name').text
                if cls not in labels:    #classes or int(difficult) == 1:
                    continue
                cls_id = labels.index(cls)
                objs['name'] = cls
                if objs['name'] in seen_labels:
                    seen_labels[objs['name']] += 1
                else:
                    seen_labels[objs['name']] = 1

                img['object'] += [objs]
                xmlbox = objnode.find('bndbox')

                objs['xmin'] = int(round(float(xmlbox.find('xmin').text)))
                objs['ymin'] = int(round(float(xmlbox.find('ymin').text)))
                objs['xmax'] = int(round(float(xmlbox.find('xmax').text)))
                objs['ymax'] = int(round(float(xmlbox.find('ymax').text)))

            if len(img['object']) > 0:
                all_insts += [img]
            # for elem in tree.iter():
            #     # if 'filename' in elem.tag:
            #     #     img['filename'] = img_dir + elem.text
            #     # if 'width' in elem.tag:
            #     #     img['width'] = int(elem.text)
            #     # if 'height' in elem.tag:
            #     #     img['height'] = int(elem.text)
            #     if 'object' in elem.tag or 'part' in elem.tag:
            #         obj = {}
            #
            #         for attr in list(elem):
            #             if 'name' in attr.tag:
            #                 obj['name'] = attr.text
            #
            #                 if obj['name'] in seen_labels:
            #                     seen_labels[obj['name']] += 1
            #                 else:
            #                     seen_labels[obj['name']] = 1
            #
            #                 if len(labels) > 0 and obj['name'] not in labels:
            #                     break
            #                 else:
            #                     img['object'] += [obj]
            #
            #             if 'bndbox' in attr.tag:
            #                 for dim in list(attr):
            #                     if 'xmin' in dim.tag:
            #                         obj['xmin'] = int(round(float(dim.text)))
            #                     if 'ymin' in dim.tag:
            #                         obj['ymin'] = int(round(float(dim.text)))
            #                     if 'xmax' in dim.tag:
            #                         obj['xmax'] = int(round(float(dim.text)))
            #                     if 'ymax' in dim.tag:
            #                         obj['ymax'] = int(round(float(dim.text)))
            #
            # if len(img['object']) > 0:
            #     all_imgs += [img]
            cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
            # with open(cache_name, 'wb') as handle:
            #     pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels

if __name__=='__main__':
    train_imgs, train_labels=parse_voc_annotation("D:/project/VOC2007/Annotations/","D:/project/VOC2007/JPEGImages/","kangaroo_train.pkl",["mouse3"])
    print(train_labels)