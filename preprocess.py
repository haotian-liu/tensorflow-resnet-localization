import argparse
import utils
import os
import errno
import numpy as np
from PIL import Image
from scipy.misc import imresize
from loader import Loader

parser = argparse.ArgumentParser()

parser.add_argument('--resize', action='store_true')

args = parser.parse_args()

if args.resize:
    with open(utils.path('data/images.txt')) as f:
        id_to_path = dict([l.split(' ', 1) for l in f.read().splitlines()])

    ids = id_to_path.keys()
    with open(utils.path('data/bounding_boxes.txt')) as f:
        id_to_box = dict()
        for line in f.read().splitlines():
            im_id, *box = line.split(' ')
            id_to_box[im_id] = list(map(float, box))
    imgs = [(i,
             os.path.join(utils.path('data/images'), id_to_path[i]),
             os.path.join(utils.path('data/resize'), id_to_path[i])) for i in ids]

    for img in imgs:
        i, ori_path, resize_path = img

        if os.path.isfile(resize_path):
            print("File %s already exists!" %i)
            continue

        im = Image.open(ori_path).convert('RGB')
        im = imresize(im, (224, 224))

        if not os.path.exists(os.path.dirname(resize_path)):
            try:
                os.makedirs(os.path.dirname(resize_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        result = Image.fromarray(im.astype(np.uint8))
        result.save(resize_path, quality=100)

        print("Saving File %s", i)