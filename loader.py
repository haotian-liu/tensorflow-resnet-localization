import os
import random, math
import tensorflow as tf
import numpy as np
import _pickle as pickle
from PIL import Image
import utils

class CUB_Dataset(object):
    def __init__(self, path, im_ids=[], transform=None):
        self.path = path
        with open(self.path + '/images.txt') as f:
            id_to_path = dict([l.split(' ', 1) for l in f.read().splitlines()])
        with open(self.path + '/bounding_boxes.txt') as f:
            id_to_box = dict()
            for line in f.read().splitlines():
                im_id, *box = line.split(' ')
                id_to_box[im_id] = list(map(float, box))
        self.imgs = [(os.path.join(self.path + '/images', id_to_path[i]), id_to_box[i]) for i in im_ids]
        if transform is None:
            self.transform = [
                lambda x: tf.image.resize_images(x, (224, 224)),
                lambda x: (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225],
            ]
        else:
            self.transform = transform

    def get_imgs_np(self):
        imgs = np.array(list(range(len(self.imgs))))
        return imgs

    def __getitem__(self, index):
        path, box = self.imgs[index]
        im = Image.open(path).convert('RGB')
        im_size = np.array(im.size, dtype='float32')
        box = np.array(box, dtype='float32')

        for f in self.transform:
            im = f(im)

        return im, box, im_size

    def __len__(self):
        return len(self.imgs)

class Loader(object):

    def __init__(self, base_path=None, path=None):
        if base_path is None:
            self.path = os.path.dirname(os.path.abspath(__file__))
        else:
            self.path = base_path

        if path is not None:
            self.path += path

        self.imgs, self.transform = None, None

    def CUB(self, ratio, tot_ratio=1.0):
        # pickle_path = utils.base_path() + "/data/datasets.pkl"
        # if os.path.isfile(pickle_path):
        #     print("Using pickled data!")
        #     return pickle.load(open(pickle_path, 'rb'))
        train_id, test_id = self.split(ratio, tot_ratio)
        splits = {'train': train_id, 'test': test_id}
        datasets = {split: CUB_Dataset(self.path, splits[split]) for split in ('train', 'test')}
        # pickle.dump(datasets, open(pickle_path, 'wb'))
        # print("Data loaded from disk and has been pickled!")
        return datasets

    def split(self, ratio, total_ratio=1.0):
        with open(self.path + '/images.txt') as f:
            lines = f.read().splitlines()
        class_groups = dict()
        for line in lines:
            value, line = line.split(' ', 1)
            key = line.split('.', 1)[0]
            value = value
            if key in class_groups:
                class_groups[key].append(value)
            else:
                class_groups[key] = [value]

        tot_id = [None] * len(class_groups)
        tot_ids = []
        index = 0
        for _, group in class_groups.items():
            tot_id[index] = []
            tot_id[index].extend(random.sample(group, int(math.ceil(len(group)*total_ratio))))
            tot_ids.extend(tot_id[index])
            index += 1

        test_id = []
        for ids in tot_id:
            test_id.extend(random.sample(ids, int(math.ceil(len(ids)*ratio))))
        train_id = [i for i in tot_ids if i not in test_id]

        return train_id, test_id