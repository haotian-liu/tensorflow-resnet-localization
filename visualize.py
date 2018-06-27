import os
import time
import utils
import random
import numpy as np
import tensorflow as tf
import cv2
from loader import Loader, CUB_Dataset
from PIL import Image
from resnet import Resnet18
from matplotlib import pyplot as plt
from BatchLoader import BatchLoader
import json

def imshow(fig, img, gt_box, pred_box=None):
    fig.imshow(img)

    def draw_box(box, color='green'):
        x, y, w, h = utils.box_transform_inv(box, img.shape[:2][::-1])[0]
        if x == 0:
            x = 1
        if y == 0:
            y = 1
        fig.add_patch(
            plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2, alpha=0.8)
        )

    draw_box(gt_box)
    if pred_box is not None:
        draw_box(pred_box, 'red')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('log', None, 'Specify log path.')
flags.DEFINE_string('model', None, 'Specify model path.')
flags.DEFINE_boolean('log_only', False, 'Whether to skip demo.')
flags.DEFINE_boolean('demo_only', False, 'Whether to skip log.')

if FLAGS.log_only:
    log = json.load(open(utils.path("logs/" + FLAGS.log), "r"))
    log = {phase: {
        stat_type: [x[stat_type] for x in log[phase]] for stat_type in ['loss', 'accu']
    } for phase in log}
    for phase in log:
        log[phase]['err'] = [1.0-x for x in log[phase]['accu']]

    # Plot statistics
    tags = ('loss', 'err')
    phases = ('train', 'test')
    sub_line = len(tags)
    sub_lines = sub_line * len(phases)
    fig = plt.figure(figsize=(12, 1.5 * sub_lines))
    sub_id = 0
    for phase in phases:
        for tag in tags:
            sub_id = sub_id + 1
            sub = fig.add_subplot(sub_lines, 1, sub_id)
            sub.plot(range(len(log[phase][tag])), log[phase][tag], label=(phase + '_' + tag))
            sub.set_title(phase + ': ' + tag)
    fig.tight_layout()
    plt.show()

# Visualize predicting result
if FLAGS.demo_only:
    figs_x, figs_y = (5, 5)
    loader = Loader(base_path=None, path="/data")
    datasets = loader.CUB(ratio=0.2, total_ratio=1.0)
    model = Resnet18(batch_size=figs_x*figs_y)
    with model.graph.as_default():
        model.preload()

    with tf.Session(graph=model.graph) as sess:
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(utils.path("models/" + FLAGS.model + "/")))
        data_loader = BatchLoader(datasets['test'], batch_size=figs_x * figs_y, pre_fetch=1,
                                  shuffle=True, op_fn=CUB_Dataset.list_to_tuple)
        fig = plt.figure(figsize=(6 * figs_x, 2 * figs_y))

        max_iter = 100
        for features, boxes, im_sizes in data_loader:
            if not max_iter or not plt.fignum_exists(fig.number):
                break
            max_iter -= 1

            fig.clf()

            boxes = utils.crop_boxes(boxes, im_sizes)
            boxes = utils.box_transform(boxes, im_sizes)

            pred_boxes = sess.run(model.fc, feed_dict={
                'features:0': features,
                'boxes:0': boxes,
                'training:0': False,
            })

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            ori_ims = std * features + mean
            ori_ims = np.clip(ori_ims, 0, 1)

            IoUs = utils.compute_accu_items(pred_boxes, boxes, im_sizes)

            for sub_id in range(figs_x * figs_y):
                sub = fig.add_subplot(figs_x, figs_y, sub_id+1)
                im, box, im_size, ori_im = features[sub_id], boxes[sub_id], im_sizes[sub_id], ori_ims[sub_id]
                pred_box, IoU = pred_boxes[sub_id], IoUs[sub_id]
                sub.set_title(("Correct" if IoU>=0.75 else "Incorrect") + ", IoU: " + str(IoU),
                              color='black' if IoU>=0.75 else 'red')
                sub.axis('off')
                imshow(sub, ori_im, box, pred_box)
            fig.tight_layout()
            plt.draw()
            plt.waitforbuttonpress(0)
        del data_loader
