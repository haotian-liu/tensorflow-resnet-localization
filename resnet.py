import tensorflow as tf

class Block(object):

    def __init__(self, filters, kernel_size, strides=1, training=False):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.training = training

    def g(self, net, name):
        conv_1 = self._g(net, name + "_1")
        self.strides = 1
        conv_2 = self._g(conv_1, name + "_2")
        return conv_2

    def _g(self, net, name):
        with tf.variable_scope(name):
            if self.strides != 1:
                residual = tf.layers.conv2d(net, self.filters, 1, strides=self.strides,
                                            padding="SAME", name="shortcut", use_bias=False)
                # residual = tf.layers.batch_normalization(residual, training=self.training)
            else:
                residual = net

            conv1 = tf.layers.conv2d(net, self.filters, self.kernel_size, strides=self.strides,
                                     activation=tf.nn.relu, padding="SAME", name="conv_1",
                                     use_bias=False)
            bn1 = tf.layers.batch_normalization(conv1, training=self.training, name="bn_1")

            conv2 = tf.layers.conv2d(bn1, self.filters, self.kernel_size, strides=1,
                                     activation=tf.nn.relu, padding="SAME", name="conv_2",
                                     use_bias=False)
            bn2 = tf.layers.batch_normalization(conv2, training=self.training, name="bn_2")

            out = tf.nn.relu(bn2 + residual)

        return out

class Resnet18(object):
    def __init__(self, mode, batch_size=32):
        self.mode = mode
        self.training = (mode == "train")
        self.graph = tf.Graph()

        self.batch_size = batch_size
        self.fc, self.loss, self.mean_loss = None, None, None

    def block(self, filters, strides):
        return Block(filters=filters, kernel_size=3, strides=strides, training=self.training)

    def preload(self):
        features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 224, 224, 3], name="features")

        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(features, 64, 7, strides=2, activation=tf.nn.relu,
                                     padding="SAME", name="conv", use_bias=False)
            bn1 = tf.layers.batch_normalization(conv1, training=self.training, name="bn")
            pool1 = tf.layers.max_pooling2d(bn1, 3, 2, padding="SAME")

        conv2 = self.block(filters=64, strides=1).g(pool1, name="conv2")
        conv3 = self.block(filters=128, strides=2).g(conv2, name="conv3")
        conv4 = self.block(filters=256, strides=2).g(conv3, name="conv4")
        conv5 = self.block(filters=512, strides=2).g(conv4, name="conv5")

        pool = tf.layers.average_pooling2d(inputs=conv5, pool_size=7, strides=1)
        fc = tf.layers.dense(inputs=pool, units=4)

        boxes = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 4], name="boxes")

        loss_raw = tf.abs(fc - boxes)
        loss = tf.where(loss_raw < 1, 0.5 * loss_raw ** 2, loss_raw - 0.5)
        # loss = tf.reduce_sum(loss, axis=3, keepdims=True)
        loss = tf.reduce_sum(loss)

        # loss = tf.nn.l2_loss(fc - boxes)

        self.mean_loss = tf.reduce_mean(loss)

        self.fc, self.loss = fc, loss