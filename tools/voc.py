import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes = ["mouse"]

def get_boxs(xml_path,labels=["mouse"]):
    seen_labels = {}

    tree = ET.parse(xml_path)
    # tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    boxs=[]
    for objnode in root.iter('object'):
        objs = {}
        difficult = 0  # objnode.find('difficult').text
        cls = objnode.find('name').text
        if cls not in labels:  # classes or int(difficult) == 1:
            continue
        cls_id = labels.index(cls)
        objs['name'] = cls
        if objs['name'] in seen_labels:
            seen_labels[objs['name']] += 1
        else:
            seen_labels[objs['name']] = 1

        if objnode.find('difficult') is None or str(objnode.find('difficult').text) == "0":
            xmlbox = objnode.find('bndbox')
            xmin = int(round(float(xmlbox.find('xmin').text)))
            ymin = int(round(float(xmlbox.find('ymin').text)))
            xmax = int(round(float(xmlbox.find('xmax').text)))
            ymax = int(round(float(xmlbox.find('ymax').text)))
            boxs.append([xmin,ymin,xmax-xmin,ymax-ymin])
    return boxs

def parse_voc_annotation(ann_dir, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}

        for xml_file in sorted(os.listdir(ann_dir)):
            img = {'object':[]}
            print(xml_file)

            tree = ET.parse(ann_dir + xml_file)
            # tree = ET.parse(in_file)
            root = tree.getroot()

            img['filename'] = img_dir + root.find('filename').text
            size = root.find('size')

            img['width'] = int(size.find('width').text)
            img['height'] =  int(size.find('height').text)
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for objnode in root.iter('object'):
                objs = {}
                difficult = 0  #objnode.find('difficult').text
                cls = objnode.find('name').text
                if cls not in labels:
                    print("error", cls, xml_file)
                    continue
                cls_id = labels.index(cls)
                objs['name'] = cls
                if objs['name'] in seen_labels:
                    seen_labels[objs['name']] += 1
                else:
                    seen_labels[objs['name']] = 1

                img['object'] += [objs]
                if objnode.find('difficult') is None or str(objnode.find('difficult').text) == "0":
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
            with open(cache_name, 'wb') as handle:
                pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels

def parse_voc_annotation_new(labels=[],org_path=None):
    # if os.path.exists(cache_name):
    #     with open(cache_name, 'rb') as handle:
    #         cache = pickle.load(handle)
    #     all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    if 1:
        all_insts = []
        seen_labels = {}

        all_files = [list(map(lambda x: os.path.join(root, x), files)) for root, _, files in
                     os.walk(org_path, topdown=False) if os.path.basename(root) == 'Annotations']
        xml_files = []
        for i in range(len(all_files)):
            xml_files += all_files[i]
        # jpg_files = [file.replace('Annotations', 'JPEGImages').replace('xml', 'jpg') for file in xml_files]
        # xml_files = [org_path + '%s.xml' % xml_file.strip('\n') for xml_file in all_files]
        # jpg_files = [org_path + '%s.jpg' % jpg_file.strip('\n').replace('Annotations', 'JPEGImages') for jpg_file in all_files]
        print(len(xml_files))
        for xml_file in sorted(xml_files):
            img = {'object':[]}
            # print("------",xml_file)

            tree = ET.parse(xml_file)
            root = tree.getroot()

            # img['filename'] = img_dir + root.find('filename').text
            img['filename'] = xml_file.replace('Annotations', 'JPEGImages').replace('xml','jpg')
            size = root.find('size')

            img['width'] = int(size.find('width').text)
            img['height'] =  int(size.find('height').text)
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for objnode in root.iter('object'):
                objs = {}
                if objnode.find('difficult') is not None and objnode.find('difficult').text == '1':
                    continue
                cls = objnode.find('name').text
                if cls not in labels:    #classes or int(difficult) == 1:
                    print("error",cls,xml_file)
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