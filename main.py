import os
import time
import utils
import random
import numpy as np
import tensorflow as tf
from loader import Loader, CUB_Dataset
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
flags.DEFINE_integer('pre_fetch', 6, 'Specify prefetch dataset count (default=6)')
flags.DEFINE_boolean('fine_tune', False, 'Specify whether to fine tune or not.')
flags.DEFINE_string('log_path', None, 'Specify log path, default disabled.')

total_ratio = FLAGS.ratio if FLAGS.ratio != 0.05 else 0.05 if FLAGS.CPU else 1.0

print("Using %f as the total ratio, %d threads." % (total_ratio, FLAGS.max_threads))
print("Training %d epochs with a batch size of %d, pre-fetching %d batches."
      % (FLAGS.num_epochs, FLAGS.batch_size, FLAGS.pre_fetch))

pool = ThreadPool(FLAGS.max_threads)

def main(unused_argv):
    loader = Loader(base_path=None, path="/data")
    datasets = loader.CUB(ratio=0.2, total_ratio=total_ratio)
    model = Resnet18(batch_size=FLAGS.batch_size)
    with model.graph.as_default():
        model.preload()

        vars = [var for var in tf.global_variables()
                if var.name.startswith("conv")]

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(1e-3, global_step=global_step,
                                                   decay_steps=5 * int(len(datasets["train"]) / FLAGS.batch_size),
                                                   decay_rate=0.1, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grad_and_vars = opt.compute_gradients(loss=model.loss)

            if FLAGS.fine_tune:
                for index, (grad, var) in enumerate(grad_and_vars):
                    if var.op.name.startswith("dense") or var.op.name.startswith("conv5"):
                        grad_and_vars[index] = (grad * 10.0, var)

            train_op = opt.apply_gradients(grad_and_vars, global_step=global_step)
            # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            #     .minimize(loss=model.loss, global_step=global_step)

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
        if os.path.exists(utils.path("models/trained")):
            tf.train.Saver().restore(sess, utils.path("models/trained/resnet18.ckpt"))
        else:
            init_rest_vars.run()
            tf.train.Saver(vars).restore(sess, utils.path("models/init/models.ckpt"))

        from BatchLoader import BatchLoader
        LOG = utils.Log()

        for epoch in range(FLAGS.num_epochs):
            for phase in ('train', 'test'):
                dataset = datasets[phase]

                accs = utils.AverageMeter()
                losses = utils.AverageMeter()
                idxs = list(range(len(dataset)))
                random.shuffle(idxs)
                start_time = time.time()
                bar = progressbar.ProgressBar()

                for features, boxes, im_sizes in bar(BatchLoader(dataset, batch_size=FLAGS.batch_size,
                                                                 pre_fetch=FLAGS.pre_fetch,
                                                                 op_fn=CUB_Dataset.list_to_tuple)):
                    boxes = utils.crop_boxes(boxes, im_sizes)
                    boxes = utils.box_transform(boxes, im_sizes)
                    boxes = np.reshape(boxes, [-1, 1, 1, 4])

                    if phase == 'train':
                        _, loss, outputs = sess.run([train_op, model.loss, model.fc], feed_dict={
                            'features:0': features,
                            'boxes:0': boxes,
                            # 'training:0': phase == 'train',
                        })
                    else:
                        loss, outputs = sess.run([model.loss, model.fc], feed_dict={
                            'features:0': features,
                            'boxes:0': boxes,
                            # 'training:0': phase == 'train',
                        })

                    outputs = np.reshape(outputs, [-1, 4])
                    boxes = np.reshape(boxes, [-1, 4])

                    acc = utils.compute_acc(outputs, boxes, im_sizes)

                    nsample = model.batch_size
                    accs.update(acc, nsample)
                    losses.update(loss, nsample)

                    LOG.add(phase, {"accu": acc, "loss": loss})

                elapsed_time = time.time() - start_time
                print('[{}]\tEpoch: {}/{}\tLoss: {:.4f}\tAcc: {:.2%}\tTime: {:.3f}'.format(
                    phase, epoch, FLAGS.num_epochs, losses.avg, accs.avg, elapsed_time))

        tf.train.Saver().save(sess, utils.path("models/trained/resnet18.ckpt"), global_step=global_step)
        if FLAGS.log_path is not None:
            LOG.dump(FLAGS.log_path)

if __name__ == "__main__":
    tf.app.run()