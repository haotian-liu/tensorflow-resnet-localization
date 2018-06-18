import os
import tensorflow as tf
import numpy as np
import _pickle as pickle
import torchfile  # pip install torchfile
import utils
from resnet import Resnet18

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # set an available GPU

# FLAGS(?)
T7_PATH = utils.path('models/resnet-18.t7')
INIT_CHECKPOINT_DIR = utils.path('models/init_t7')


# Open ResNet-18 torch checkpoint
print('Open ResNet-18 torch checkpoint: %s' % T7_PATH)
o = torchfile.load(T7_PATH)

# Load weights in a brute-force way
print('Load weights in a brute-force way')
modules = o.modules
conv1_weights = o.modules[0].weight
conv1_bn_gamma = o.modules[1].weight
conv1_bn_beta = o.modules[1].bias
conv1_bn_mean = o.modules[1].running_mean
conv1_bn_var = o.modules[1].running_var

conv2_1_weights_1  = o.modules[4].modules[0].modules[0].modules[0].modules[0].weight
conv2_1_bn_1_gamma = o.modules[4].modules[0].modules[0].modules[0].modules[1].weight
conv2_1_bn_1_beta  = o.modules[4].modules[0].modules[0].modules[0].modules[1].bias
conv2_1_bn_1_mean  = o.modules[4].modules[0].modules[0].modules[0].modules[1].running_mean
conv2_1_bn_1_var   = o.modules[4].modules[0].modules[0].modules[0].modules[1].running_var
conv2_1_weights_2  = o.modules[4].modules[0].modules[0].modules[0].modules[3].weight
conv2_1_bn_2_gamma = o.modules[4].modules[0].modules[0].modules[0].modules[4].weight
conv2_1_bn_2_beta  = o.modules[4].modules[0].modules[0].modules[0].modules[4].bias
conv2_1_bn_2_mean  = o.modules[4].modules[0].modules[0].modules[0].modules[4].running_mean
conv2_1_bn_2_var   = o.modules[4].modules[0].modules[0].modules[0].modules[4].running_var
conv2_2_weights_1  = o.modules[4].modules[1].modules[0].modules[0].modules[0].weight
conv2_2_bn_1_gamma = o.modules[4].modules[1].modules[0].modules[0].modules[1].weight
conv2_2_bn_1_beta  = o.modules[4].modules[1].modules[0].modules[0].modules[1].bias
conv2_2_bn_1_mean  = o.modules[4].modules[1].modules[0].modules[0].modules[1].running_mean
conv2_2_bn_1_var   = o.modules[4].modules[1].modules[0].modules[0].modules[1].running_var
conv2_2_weights_2  = o.modules[4].modules[1].modules[0].modules[0].modules[3].weight
conv2_2_bn_2_gamma = o.modules[4].modules[1].modules[0].modules[0].modules[4].weight
conv2_2_bn_2_beta  = o.modules[4].modules[1].modules[0].modules[0].modules[4].bias
conv2_2_bn_2_mean  = o.modules[4].modules[1].modules[0].modules[0].modules[4].running_mean
conv2_2_bn_2_var   = o.modules[4].modules[1].modules[0].modules[0].modules[4].running_var

conv3_1_weights_skip = o.modules[5].modules[0].modules[0].modules[1].weight
conv3_1_weights_1  = o.modules[5].modules[0].modules[0].modules[0].modules[0].weight
conv3_1_bn_1_gamma = o.modules[5].modules[0].modules[0].modules[0].modules[1].weight
conv3_1_bn_1_beta  = o.modules[5].modules[0].modules[0].modules[0].modules[1].bias
conv3_1_bn_1_mean  = o.modules[5].modules[0].modules[0].modules[0].modules[1].running_mean
conv3_1_bn_1_var   = o.modules[5].modules[0].modules[0].modules[0].modules[1].running_var
conv3_1_weights_2  = o.modules[5].modules[0].modules[0].modules[0].modules[3].weight
conv3_1_bn_2_gamma = o.modules[5].modules[0].modules[0].modules[0].modules[4].weight
conv3_1_bn_2_beta  = o.modules[5].modules[0].modules[0].modules[0].modules[4].bias
conv3_1_bn_2_mean  = o.modules[5].modules[0].modules[0].modules[0].modules[4].running_mean
conv3_1_bn_2_var   = o.modules[5].modules[0].modules[0].modules[0].modules[4].running_var
conv3_2_weights_1  = o.modules[5].modules[1].modules[0].modules[0].modules[0].weight
conv3_2_bn_1_gamma = o.modules[5].modules[1].modules[0].modules[0].modules[1].weight
conv3_2_bn_1_beta  = o.modules[5].modules[1].modules[0].modules[0].modules[1].bias
conv3_2_bn_1_mean  = o.modules[5].modules[1].modules[0].modules[0].modules[1].running_mean
conv3_2_bn_1_var   = o.modules[5].modules[1].modules[0].modules[0].modules[1].running_var
conv3_2_weights_2  = o.modules[5].modules[1].modules[0].modules[0].modules[3].weight
conv3_2_bn_2_gamma = o.modules[5].modules[1].modules[0].modules[0].modules[4].weight
conv3_2_bn_2_beta  = o.modules[5].modules[1].modules[0].modules[0].modules[4].bias
conv3_2_bn_2_mean  = o.modules[5].modules[1].modules[0].modules[0].modules[4].running_mean
conv3_2_bn_2_var   = o.modules[5].modules[1].modules[0].modules[0].modules[4].running_var

conv4_1_weights_skip = o.modules[6].modules[0].modules[0].modules[1].weight
conv4_1_weights_1  = o.modules[6].modules[0].modules[0].modules[0].modules[0].weight
conv4_1_bn_1_gamma = o.modules[6].modules[0].modules[0].modules[0].modules[1].weight
conv4_1_bn_1_beta  = o.modules[6].modules[0].modules[0].modules[0].modules[1].bias
conv4_1_bn_1_mean  = o.modules[6].modules[0].modules[0].modules[0].modules[1].running_mean
conv4_1_bn_1_var   = o.modules[6].modules[0].modules[0].modules[0].modules[1].running_var
conv4_1_weights_2  = o.modules[6].modules[0].modules[0].modules[0].modules[3].weight
conv4_1_bn_2_gamma = o.modules[6].modules[0].modules[0].modules[0].modules[4].weight
conv4_1_bn_2_beta  = o.modules[6].modules[0].modules[0].modules[0].modules[4].bias
conv4_1_bn_2_mean  = o.modules[6].modules[0].modules[0].modules[0].modules[4].running_mean
conv4_1_bn_2_var   = o.modules[6].modules[0].modules[0].modules[0].modules[4].running_var
conv4_2_weights_1  = o.modules[6].modules[1].modules[0].modules[0].modules[0].weight
conv4_2_bn_1_gamma = o.modules[6].modules[1].modules[0].modules[0].modules[1].weight
conv4_2_bn_1_beta  = o.modules[6].modules[1].modules[0].modules[0].modules[1].bias
conv4_2_bn_1_mean  = o.modules[6].modules[1].modules[0].modules[0].modules[1].running_mean
conv4_2_bn_1_var   = o.modules[6].modules[1].modules[0].modules[0].modules[1].running_var
conv4_2_weights_2  = o.modules[6].modules[1].modules[0].modules[0].modules[3].weight
conv4_2_bn_2_gamma = o.modules[6].modules[1].modules[0].modules[0].modules[4].weight
conv4_2_bn_2_beta  = o.modules[6].modules[1].modules[0].modules[0].modules[4].bias
conv4_2_bn_2_mean  = o.modules[6].modules[1].modules[0].modules[0].modules[4].running_mean
conv4_2_bn_2_var   = o.modules[6].modules[1].modules[0].modules[0].modules[4].running_var

conv5_1_weights_skip = o.modules[7].modules[0].modules[0].modules[1].weight
conv5_1_weights_1  = o.modules[7].modules[0].modules[0].modules[0].modules[0].weight
conv5_1_bn_1_gamma = o.modules[7].modules[0].modules[0].modules[0].modules[1].weight
conv5_1_bn_1_beta  = o.modules[7].modules[0].modules[0].modules[0].modules[1].bias
conv5_1_bn_1_mean  = o.modules[7].modules[0].modules[0].modules[0].modules[1].running_mean
conv5_1_bn_1_var   = o.modules[7].modules[0].modules[0].modules[0].modules[1].running_var
conv5_1_weights_2  = o.modules[7].modules[0].modules[0].modules[0].modules[3].weight
conv5_1_bn_2_gamma = o.modules[7].modules[0].modules[0].modules[0].modules[4].weight
conv5_1_bn_2_beta  = o.modules[7].modules[0].modules[0].modules[0].modules[4].bias
conv5_1_bn_2_mean  = o.modules[7].modules[0].modules[0].modules[0].modules[4].running_mean
conv5_1_bn_2_var   = o.modules[7].modules[0].modules[0].modules[0].modules[4].running_var
conv5_2_weights_1  = o.modules[7].modules[1].modules[0].modules[0].modules[0].weight
conv5_2_bn_1_gamma = o.modules[7].modules[1].modules[0].modules[0].modules[1].weight
conv5_2_bn_1_beta  = o.modules[7].modules[1].modules[0].modules[0].modules[1].bias
conv5_2_bn_1_mean  = o.modules[7].modules[1].modules[0].modules[0].modules[1].running_mean
conv5_2_bn_1_var   = o.modules[7].modules[1].modules[0].modules[0].modules[1].running_var
conv5_2_weights_2  = o.modules[7].modules[1].modules[0].modules[0].modules[3].weight
conv5_2_bn_2_gamma = o.modules[7].modules[1].modules[0].modules[0].modules[4].weight
conv5_2_bn_2_beta  = o.modules[7].modules[1].modules[0].modules[0].modules[4].bias
conv5_2_bn_2_mean  = o.modules[7].modules[1].modules[0].modules[0].modules[4].running_mean
conv5_2_bn_2_var   = o.modules[7].modules[1].modules[0].modules[0].modules[4].running_var

fc_weights = o.modules[10].weight
fc_biases = o.modules[10].bias

model_weights_temp = {
    'conv1/conv/kernel': conv1_weights,
    'conv1/bn/moving_mean': conv1_bn_mean,
    'conv1/bn/moving_variance': conv1_bn_var,
    'conv1/bn/beta': conv1_bn_beta,
    'conv1/bn/gamma': conv1_bn_gamma,

    'conv2_1/conv_1/kernel': conv2_1_weights_1,
    'conv2_1/bn_1/moving_mean':       conv2_1_bn_1_mean,
    'conv2_1/bn_1/moving_variance':    conv2_1_bn_1_var,
    'conv2_1/bn_1/beta':     conv2_1_bn_1_beta,
    'conv2_1/bn_1/gamma':    conv2_1_bn_1_gamma,
    'conv2_1/conv_2/kernel': conv2_1_weights_2,
    'conv2_1/bn_2/moving_mean':       conv2_1_bn_2_mean,
    'conv2_1/bn_2/moving_variance':    conv2_1_bn_2_var,
    'conv2_1/bn_2/beta':     conv2_1_bn_2_beta,
    'conv2_1/bn_2/gamma':    conv2_1_bn_2_gamma,
    'conv2_2/conv_1/kernel': conv2_2_weights_1,
    'conv2_2/bn_1/moving_mean':       conv2_2_bn_1_mean,
    'conv2_2/bn_1/moving_variance':    conv2_2_bn_1_var,
    'conv2_2/bn_1/beta':     conv2_2_bn_1_beta,
    'conv2_2/bn_1/gamma':    conv2_2_bn_1_gamma,
    'conv2_2/conv_2/kernel': conv2_2_weights_2,
    'conv2_2/bn_2/moving_mean':       conv2_2_bn_2_mean,
    'conv2_2/bn_2/moving_variance':    conv2_2_bn_2_var,
    'conv2_2/bn_2/beta':     conv2_2_bn_2_beta,
    'conv2_2/bn_2/gamma':    conv2_2_bn_2_gamma,

    'conv3_1/shortcut/kernel':  conv3_1_weights_skip,
    'conv3_1/conv_1/kernel': conv3_1_weights_1,
    'conv3_1/bn_1/moving_mean':       conv3_1_bn_1_mean,
    'conv3_1/bn_1/moving_variance':    conv3_1_bn_1_var,
    'conv3_1/bn_1/beta':     conv3_1_bn_1_beta,
    'conv3_1/bn_1/gamma':    conv3_1_bn_1_gamma,
    'conv3_1/conv_2/kernel': conv3_1_weights_2,
    'conv3_1/bn_2/moving_mean':       conv3_1_bn_2_mean,
    'conv3_1/bn_2/moving_variance':    conv3_1_bn_2_var,
    'conv3_1/bn_2/beta':     conv3_1_bn_2_beta,
    'conv3_1/bn_2/gamma':    conv3_1_bn_2_gamma,
    'conv3_2/conv_1/kernel': conv3_2_weights_1,
    'conv3_2/bn_1/moving_mean':       conv3_2_bn_1_mean,
    'conv3_2/bn_1/moving_variance':    conv3_2_bn_1_var,
    'conv3_2/bn_1/beta':     conv3_2_bn_1_beta,
    'conv3_2/bn_1/gamma':    conv3_2_bn_1_gamma,
    'conv3_2/conv_2/kernel': conv3_2_weights_2,
    'conv3_2/bn_2/moving_mean':       conv3_2_bn_2_mean,
    'conv3_2/bn_2/moving_variance':    conv3_2_bn_2_var,
    'conv3_2/bn_2/beta':     conv3_2_bn_2_beta,
    'conv3_2/bn_2/gamma':    conv3_2_bn_2_gamma,

    'conv4_1/shortcut/kernel':  conv4_1_weights_skip,
    'conv4_1/conv_1/kernel': conv4_1_weights_1,
    'conv4_1/bn_1/moving_mean':       conv4_1_bn_1_mean,
    'conv4_1/bn_1/moving_variance':    conv4_1_bn_1_var,
    'conv4_1/bn_1/beta':     conv4_1_bn_1_beta,
    'conv4_1/bn_1/gamma':    conv4_1_bn_1_gamma,
    'conv4_1/conv_2/kernel': conv4_1_weights_2,
    'conv4_1/bn_2/moving_mean':       conv4_1_bn_2_mean,
    'conv4_1/bn_2/moving_variance':    conv4_1_bn_2_var,
    'conv4_1/bn_2/beta':     conv4_1_bn_2_beta,
    'conv4_1/bn_2/gamma':    conv4_1_bn_2_gamma,
    'conv4_2/conv_1/kernel': conv4_2_weights_1,
    'conv4_2/bn_1/moving_mean':       conv4_2_bn_1_mean,
    'conv4_2/bn_1/moving_variance':    conv4_2_bn_1_var,
    'conv4_2/bn_1/beta':     conv4_2_bn_1_beta,
    'conv4_2/bn_1/gamma':    conv4_2_bn_1_gamma,
    'conv4_2/conv_2/kernel': conv4_2_weights_2,
    'conv4_2/bn_2/moving_mean':       conv4_2_bn_2_mean,
    'conv4_2/bn_2/moving_variance':    conv4_2_bn_2_var,
    'conv4_2/bn_2/beta':     conv4_2_bn_2_beta,
    'conv4_2/bn_2/gamma':    conv4_2_bn_2_gamma,

    'conv5_1/shortcut/kernel':  conv5_1_weights_skip,
    'conv5_1/conv_1/kernel': conv5_1_weights_1,
    'conv5_1/bn_1/moving_mean':       conv5_1_bn_1_mean,
    'conv5_1/bn_1/moving_variance':    conv5_1_bn_1_var,
    'conv5_1/bn_1/beta':     conv5_1_bn_1_beta,
    'conv5_1/bn_1/gamma':    conv5_1_bn_1_gamma,
    'conv5_1/conv_2/kernel': conv5_1_weights_2,
    'conv5_1/bn_2/moving_mean':       conv5_1_bn_2_mean,
    'conv5_1/bn_2/moving_variance':    conv5_1_bn_2_var,
    'conv5_1/bn_2/beta':     conv5_1_bn_2_beta,
    'conv5_1/bn_2/gamma':    conv5_1_bn_2_gamma,
    'conv5_2/conv_1/kernel': conv5_2_weights_1,
    'conv5_2/bn_1/moving_mean':       conv5_2_bn_1_mean,
    'conv5_2/bn_1/moving_variance':    conv5_2_bn_1_var,
    'conv5_2/bn_1/beta':     conv5_2_bn_1_beta,
    'conv5_2/bn_1/gamma':    conv5_2_bn_1_gamma,
    'conv5_2/conv_2/kernel': conv5_2_weights_2,
    'conv5_2/bn_2/moving_mean':       conv5_2_bn_2_mean,
    'conv5_2/bn_2/moving_variance':    conv5_2_bn_2_var,
    'conv5_2/bn_2/beta':     conv5_2_bn_2_beta,
    'conv5_2/bn_2/gamma':    conv5_2_bn_2_gamma,

    'logits/fc/weights': fc_weights,
    'logits/fc/biases': fc_biases,
}

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
    # print('Number of Weights: %d' % network_train._weights)
    # print('FLOPs: %d' % network_train._flops)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
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
