import os
import utils
import numpy as np
import tensorflow as tf
from loader import Loader
from resnet import Resnet18

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
    loader = Loader(base_path=None, path="/data")
    datasets = loader.CUB(ratio=0.2, tot_ratio=0.05)
    train_set = datasets["train"]
    batch_size = 32
    steps_per_epoch = int(len(train_set) / batch_size)
    model = Resnet18(mode="train", batch_size=batch_size)
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

        init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("logs/", model.graph)
    writer.flush()
    writer.close()

    # vars = [var.name for var in vars]
    # print('\n'.join(vars))
    # import sys
    # sys.exit(0)

    with tf.Session(graph=model.graph) as sess:
        init.run()
        tf.train.Saver(vars).restore(sess, utils.base_path() + "/models/init/models.ckpt")
        idx = list(range(len(train_set)))
        import random
        random.shuffle(idx)
        for step in range(steps_per_epoch - 1):
            sub_idx = idx[step * batch_size:(step + 1) * batch_size]
            images = [train_set[idx] for idx in sub_idx]

            features = np.array([x[0].eval() for x in images])
            boxes = np.array([x[1] for x in images])
            im_sizes = np.array([x[2] for x in images])
            boxes = utils.crop_boxes(boxes, im_sizes)
            boxes = utils.box_transform(boxes, im_sizes)

            boxes = np.reshape(boxes, [-1, 1, 1, 4])
            _, mean_loss = sess.run([train_op, model.mean_loss], feed_dict={
                'features:0': features,
                'boxes:0': boxes
            })

            print(step, mean_loss)

if __name__ == "__main__":
    tf.app.run()