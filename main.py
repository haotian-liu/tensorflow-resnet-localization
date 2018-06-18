import os
import time
import utils
import random
import numpy as np
import tensorflow as tf
from loader import Loader
from resnet import Resnet18
import progressbar

from multiprocessing.dummy import Pool as ThreadPool

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('CPU', False, 'If false, tot_ratio will be set to 1,'
                                   'it will be overridden by FLAGS.tot_ratio.')
flags.DEFINE_float('ratio', 0.05, 'If changed, tot_ratio will be changed.')
flags.DEFINE_integer('max_threads', 4, 'Specify max threads (default=4)')
flags.DEFINE_integer('batch_size', 32, 'Specify batch size (default=32)')
flags.DEFINE_integer('num_epochs', 20, 'Specify training epochs (default=20)')

total_ratio = FLAGS.ratio if FLAGS.ratio != 0.05 else 0.05 if FLAGS.CPU else 1.0

print("Using %f as the total ratio, %d threads." % (total_ratio, FLAGS.max_threads))
print("Training %d epochs with a batch size of %d." % (FLAGS.num_epochs, FLAGS.batch_size))

pool = ThreadPool(FLAGS.max_threads)

def main(unused_argv):
    loader = Loader(base_path=None, path="/data")
    datasets = loader.CUB(ratio=0.2, total_ratio=total_ratio)
    train_set = datasets["train"]
    steps_per_epoch = int(len(train_set) / FLAGS.batch_size)
    model = Resnet18(mode="train", batch_size=FLAGS.batch_size)
    with model.graph.as_default():
        model.preload()

        vars = [var for var in tf.global_variables()
                if var.name.startswith("conv")]

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(1e-3, global_step=global_step,
                                                   decay_steps=5 * steps_per_epoch,
                                                   decay_rate=0.1, staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(loss=model.loss, global_step=global_step)

        rest_vars = list(set([var for var in tf.global_variables()]) - set(vars))

        init_rest_vars = tf.variables_initializer(rest_vars)

    # writer = tf.summary.FileWriter("logs/", model.graph)
    # writer.flush()
    # writer.close()

    # vars = [var.name for var in vars]
    # print('\n'.join(vars))
    # import sys
    # sys.exit(0)

    with tf.Session(graph=model.graph) as sess:
        tf.train.Saver(vars).restore(sess, utils.base_path() + "/models/init/models.ckpt")
        init_rest_vars.run()

        from BatchLoader import BatchLoader

        for phase in ('train'):
            for epoch in range(FLAGS.num_epochs):
                accs = utils.AverageMeter()
                losses = utils.AverageMeter()
                idxs = list(range(len(train_set)))
                random.shuffle(idxs)
                start_time = time.time()
                bar = progressbar.ProgressBar()

                for images in bar(BatchLoader(train_set)):
                # for step in bar(range(steps_per_epoch - 1)):
                #     start_idx = step * FLAGS.batch_size
                #     end_idx = (step + 1) * FLAGS.batch_size
                #
                #     # Use multi-thread to accelerate data loading
                #     images = pool.map(lambda idx: train_set[idxs[idx]], range(start_idx, end_idx))
                    # images = [train_set[idxs[idx]]
                    #           for idx in range(start_idx, end_idx)]

                    features = np.array([x[0] for x in images])
                    boxes = np.array([x[1] for x in images])
                    im_sizes = np.array([x[2] for x in images])
                    boxes = utils.crop_boxes(boxes, im_sizes)
                    boxes = utils.box_transform(boxes, im_sizes)
                    boxes = np.reshape(boxes, [-1, 1, 1, 4])

                    _, loss, outputs = sess.run([train_op, model.loss, model.fc], feed_dict={
                        'features:0': features,
                        'boxes:0': boxes
                    })

                    outputs = np.reshape(outputs, [-1, 4])
                    boxes = np.reshape(boxes, [-1, 4])

                    acc = utils.compute_acc(outputs, boxes, im_sizes)

                    nsample = model.batch_size
                    accs.update(acc, nsample)
                    losses.update(loss, nsample)

                elapsed_time = time.time() - start_time
                print('[{}]\tEpoch: {}/{}\tLoss: {:.4f}\tAcc: {:.2%}\tTime: {:.3f}'.format(
                    phase, epoch, FLAGS.num_epochs, losses.avg, accs.avg, elapsed_time))

if __name__ == "__main__":
    tf.app.run()