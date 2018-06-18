import tensorflow as tf
import numpy as np
import _pickle as pickle
from resnet import Resnet18
import utils
import os

PKL_PATH = utils.path('models/resnet-18.pkl')
INIT_CHECKPOINT_DIR = utils.path('models/init')
model_weights_temp = pickle.load(open(PKL_PATH, "rb"))

# Transpose conv and fc weights
model_weights = {}
for k, v in model_weights_temp.items():
    if len(v.shape) == 4:
        model_weights[k] = np.transpose(v, (2, 3, 1, 0))
    elif len(v.shape) == 2:
        model_weights[k] = np.transpose(v)
    else:
        model_weights[k] = v


# Build ResNet-18 models and save parameters
# Build models
print("Build ResNet-18 models")
model = Resnet18(mode="train", batch_size=32)
with model.graph.as_default():
    model.preload()
with tf.Session(graph=model.graph) as sess:
    init = tf.global_variables_initializer()

    sess.run(init)

    # Set variables values
    print('Set variables to loaded weights')
    all_vars = tf.global_variables()
    for v in all_vars:
        if v.op.name == 'global_step':
            continue
        if v.op.name.startswith('dense'):
            continue
        print('\t' + v.op.name)
        assign_op = v.assign(model_weights[v.op.name])
        sess.run(assign_op)

    # Save as checkpoint
    print('Save as checkpoint: %s' % INIT_CHECKPOINT_DIR)
    if not os.path.exists(INIT_CHECKPOINT_DIR):
        os.mkdir(INIT_CHECKPOINT_DIR)
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, os.path.join(INIT_CHECKPOINT_DIR, 'models.ckpt'))

print('Done!')
