import random
import numpy as np

from tools.voc import parse_voc_annotation
import json
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid
        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.
    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids):
    out_string = ''

    anchors = centroids.copy()
    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)
    for i in sorted_indices:
        out_string += str(int(anchors[i,0]*416)) + ',' + str(int(anchors[i,1]*416)) + ', '
            
    print("old",out_string[:-2])

def run_kmeans_old(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        # print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(ann_dims, centroids))
            print_anchors(centroids)
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def run_kmeans(ann_dims, num_anchors):

    input_x = ann_dims

    a = []
    for k in range(num_anchors, 13, 1):
        kmeans = KMeans(n_clusters=k, init='k-means++',
                        n_init=10,
                        max_iter=100000,
                        tol=0.00001,
                        precompute_distances='auto',
                        verbose=0,
                        random_state=None,
                        copy_x=True,
                        n_jobs=1,
                        algorithm='auto')
        kmeans.fit(input_x)
        value = sum(np.min(cdist(input_x, kmeans.cluster_centers_, 'euclidean'), axis=1)) / input_x.shape[0]
        print(k, value)
        # print(kmeans.labels_)
        anchors = kmeans.cluster_centers_ * 416

        widths = anchors[:, 0]
        sorted_indices = np.argsort(widths)
        out_string = ""
        for i in sorted_indices:
            out_string += str(int(anchors[i, 0])) + ',' + str(int(anchors[i, 1])) + ', '
        print("new", out_string[:-2])
        a.append(value)

    cha = [a[i] - a[i + 1] for i in range(len(a) - 1)]
    a_v = a[cha.index(max(cha)) + 1]
    index = a.index(a_v) + 1
    print(max(cha), a_v, index)


def get_anchor(config,num_anchors):
    train_imgs, train_labels = parse_voc_annotation(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['model']['labels']
    )
    # run k_mean to find the anchors
    annotation_dims = []
    for image in train_imgs:
        for obj in image['object']:
            relative_w = (float(obj['xmax']) - float(obj['xmin']))/image['width']
            relatice_h = (float(obj["ymax"]) - float(obj['ymin']))/image['height']
            annotation_dims.append(tuple(map(float, (relative_w,relatice_h))))

    annotation_dims = np.array(annotation_dims)
    run_kmeans(annotation_dims, num_anchors)
    # centroids = run_kmeans_old(annotation_dims, num_anchors)
    # write anchors to file


if __name__ == '__main__':

    num_anchors = 8
    vocpath = r'\\192.168.55.39\team-CV\dataset\tiny_data_0627\train\Original/'
    pkl_name="Original_train.pkl"
    config = {}
    config['train'] = {}
    config['model'] = {}
    config['train']['train_annot_folder'] = vocpath + "/Annotations/"
    config['train']['train_image_folder'] = vocpath + "/JPEGImages/"
    config['train']['cache_name'] = pkl_name
    config['model']['labels'] = "mouse"

    get_anchor(config,num_anchors)
    '''
6 0.023571154412013316
new 19,29, 26,81, 36,60, 45,41, 60,102, 74,228
7 0.021015908736880976
new 17,24, 23,40, 26,81, 38,62, 46,40, 60,103, 74,228
8 0.01948809633285192
new 16,24, 23,39, 25,85, 31,67, 43,55, 46,38, 61,102, 74,228
9 0.018829769725668605
new 16,24, 23,39, 25,84, 31,66, 42,54, 46,38, 56,81, 59,121, 74,236
10 0.017862497823133056
new 16,23, 23,36, 24,56, 25,85, 34,70, 43,54, 46,38, 59,124, 59,83, 70,240
11 0.01715094795996884
new 16,23, 23,37, 23,80, 25,56, 32,96, 36,68, 44,52, 46,37, 62,83, 70,240, 71,130
12 0.016480189858413853
new 16,23, 16,37, 23,80, 26,56, 28,35, 32,96, 36,68, 44,52, 47,37, 62,82, 70,240, 71,130
    '''