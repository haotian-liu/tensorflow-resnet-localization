import os.path
import time
import numpy as np

def base_path():
    return os.path.dirname(os.path.abspath(__file__))

def path(sub_path):
    return base_path() + "/" + sub_path

def to_2d_tensor(inp):
    inp = np.array(inp)
    if len(inp.shape) < 2:
        inp = np.expand_dims(inp, axis=0)
    return inp

def xywh_to_x1y1x2y2(boxes):
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] += boxes[:, 0] - 1
    boxes[:, 3] += boxes[:, 1] - 1
    return boxes

def x1y1x2y2_to_xywh(boxes):
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] -= boxes[:, 0] - 1
    boxes[:, 3] -= boxes[:, 1] - 1
    return boxes

def crop_boxes(boxes, im_sizes):
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes = xywh_to_x1y1x2y2(boxes)
    zero = np.array([0])
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], im_sizes[:, 0]), zero)
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], im_sizes[:, 1]), zero)
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], im_sizes[:, 0]), zero)
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], im_sizes[:, 1]), zero)
    boxes = x1y1x2y2_to_xywh(boxes)
    return boxes

def box_transform(boxes, im_sizes):
    # box in (x, y, w, h) format
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes[:, 0] = 2 * boxes[:, 0] / im_sizes[:, 0] - 1
    boxes[:, 1] = 2 * boxes[:, 1] / im_sizes[:, 1] - 1
    boxes[:, 2] = 2 * boxes[:, 2] / im_sizes[:, 0]
    boxes[:, 3] = 2 * boxes[:, 3] / im_sizes[:, 1]
    return boxes

def box_transform_inv(boxes, im_sizes):
    # box in (x, y, w, h) format
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes[:, 0] = (boxes[:, 0] + 1) / 2 * im_sizes[:, 0]
    boxes[:, 1] = (boxes[:, 1] + 1) / 2 * im_sizes[:, 1]
    boxes[:, 2] = boxes[:, 2] / 2 * im_sizes[:, 0]
    boxes[:, 3] = boxes[:, 3] / 2 * im_sizes[:, 1]
    return boxes

def compute_IoU(boxes1, boxes2):
    boxes1 = to_2d_tensor(boxes1)
    boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes2 = to_2d_tensor(boxes2)
    boxes2 = xywh_to_x1y1x2y2(boxes2)

    intersec = np.copy(boxes1)
    intersec[:, 0] = np.maximum(boxes1[:, 0], boxes2[:, 0])
    intersec[:, 1] = np.maximum(boxes1[:, 1], boxes2[:, 1])
    intersec[:, 2] = np.minimum(boxes1[:, 2], boxes2[:, 2])
    intersec[:, 3] = np.minimum(boxes1[:, 3], boxes2[:, 3])

    def compute_area(boxes):
        # in (x1, y1, x2, y2) format
        dx = boxes[:, 2] - boxes[:, 0]
        dx[dx < 0] = 0
        dy = boxes[:, 3] - boxes[:, 1]
        dy[dy < 0] = 0
        return dx * dy

    a1 = compute_area(boxes1)
    a2 = compute_area(boxes2)
    ia = compute_area(intersec)
    assert((a1 + a2 - ia <= 0).sum() == 0)

    return ia / (a1 + a2 - ia)

def compute_accu_items(preds, targets, im_sizes):
    preds = box_transform_inv(np.copy(preds), im_sizes)
    preds = crop_boxes(preds, im_sizes)
    targets = box_transform_inv(np.copy(targets), im_sizes)
    IoU = compute_IoU(preds, targets)
    return IoU

def compute_acc(preds, targets, im_sizes, theta=0.75):
    corr = (compute_accu_items(preds, targets, im_sizes, theta) >= theta).sum()
    return corr / preds.shape[0]

class Timer(object):
    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.time()

    def end(self, mark=""):
        elapsed_time = time.time() - self._start
        print('Elapsed time ' + mark + ' {:.3f}'.format(elapsed_time))
        self.reset()

    def reset(self):
        self._start = None

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class Log(object):
    def __init__(self):
        self.train_log = []
        self.test_log = []

    def add(self, log_type, log):
        logger = self.train_log if log_type == 'train' else self.test_log
        logger.append(log)

    def dump(self, dump_path):
        import json
        dump_path = path(dump_path)
        json.dump({
            "train": self.train_log,
            "test": self.test_log,
        }, open(dump_path, "w"))

    def clear(self):
        self.train_log.clear()
        self.test_log.clear()