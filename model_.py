import time
import os
import random
import math
import numpy as np
import tensorflow as tf

from utils import *

class Conv3DNet():
    def __init__(self, session, i_depth, i_height, i_width, i_cdim, num_classes, batch_size, num_epoch, model_dir, checkpoint_name,
                 lr_decay, visualize=False, learning_rate = 0.0001, learning_rate_decay_factor = 0.99, epochs_per_decay = 10, train_rate = 0.7, train_type="tvt", f_d=5, f_h=5, f_w=5, f_filter=32, beta1=0.5, forward_only=False):

        self.session = session
        self.i_depth = i_depth
        self.i_height = i_height
        self.i_width = i_width
        self.i_cdim = i_cdim
        self.num_classes = num_classes
        #self.num_data_size = num_data_size # until now, this value isn't used
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.model_dir = model_dir
        self.checkpoint_name = checkpoint_name
        self.lr_decay = lr_decay

        if lr_decay == "exponential":
            # apply exponential decay
            batches_per_epoch = self.num_epoch / self.batch_size
            decay_steps = batches_per_epoch * epochs_per_decay
            global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable = False)
            self.learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, learning_rate_decay_factor, staircase = True)
        elif lr_decay == "normal":
            self.learning_rate = tf.Variable(learning_rate, trainable=False)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.train_rate = train_rate
        self.train_type = train_type
        self.f_filter = f_filter
        self.beta1 = beta1
        self.forward_only = forward_only
        self.visualize = visualize

        # Constants dictating the learning rate schedule.
        self.RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
        self.RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
        self.RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

        self.bn_epsilon = 1e-5
        self.bn_momentum = 0.9





    def create_model(self):
        # Placeholder

        self.X = tf.placeholder(tf.float32, [None, self.i_depth, self.i_height, self.i_width, self.i_cdim])
        self.Y = tf.placeholder(tf.int32, [None])
        self.Yhat = tf.one_hot(self.Y, self.num_classes)
        self.drop_rate = tf.placeholder(tf.float32)

        # batch normalization
        self.bn0 = batch_norm(name='bn0')
        self.bn1 = batch_norm(name='bn1')
        self.bn2 = batch_norm(name='bn2')
        self.bn3 = batch_norm(name='bn3')
        self.bn4 = batch_norm(name='bn4')


        # _, self.logits = self.layer(self.X, self.i_depth, self.i_height, self.i_width, self.f_filter)
        #_, self.logits = self.layer_v4(self.X, self.i_depth, self.i_height, self.i_width, self.f_filter, self.drop_rate)
        #_, self.logits, self.featurev = self.PDNet_v0(self.X, self.i_depth, self.i_height, self.i_width, self.f_filter, self.drop_rate)
        #_, self.logits, self.featurev  = self.layer_v7(self.X, self.i_depth, self.i_height, self.i_width, self.f_filter, self.drop_rate)
        self.logits, self.featurev = self.layer_v8(self.X)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(x=self.logits, y=self.Yhat))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.loss)
        #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, self.RMSPROP_DECAY,
        #                                          momentum=self.RMSPROP_MOMENTUM,
        #                                          epsilon=self.RMSPROP_EPSILON).minimize(self.loss)


        #self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9).minimize(self.loss)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Yhat, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        self.saver = tf.train.Saver(tf.global_variables()) # saver object for created graphs


        self.load(self.model_dir)



    def load(self, model_dir):
        if os.path.isdir(model_dir) is False:
            os.makedirs(model_dir)

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path): # check whether created checkpoint_path is
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else :
            print("Create model with new parameters.")
            self.session.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()

    def layer(self, X, f_depth, f_height, f_width, f_filter): # for size of 20, 50, 50
        h0 = lrelu(self.bn0(conv3d(X, self.f_filter, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_0')))
        h1 = lrelu(self.bn1(conv3d(h0, self.f_filter*2, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_1')))
        h2 = lrelu(self.bn2(conv3d(h1, self.f_filter*4, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_2')))
        shape = h2.get_shape().as_list()
        h2 = tf.reshape(h2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        h3 = linear(h2, self.num_classes, scope='h3_lin')

        return lrelu(h3), h3

    # add dropout
    def layer_v2(self, X, f_depth, f_height, f_width, f_filter, drop_rate): # for size of 20, 256, 256
        h0 = dropout(lrelu(self.bn0(conv3d(X, self.f_filter, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_0'))), drop_rate=drop_rate)
        h1 = dropout(lrelu(self.bn1(conv3d(h0, self.f_filter * 2, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_1'))), drop_rate=drop_rate)
        h2 = dropout(lrelu(self.bn2(conv3d(h1, self.f_filter * 4, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_2'))), drop_rate=drop_rate)
        shape = h2.get_shape().as_list()
        h2 = tf.reshape(h2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        h3 = dropout(linear(h2, self.num_classes, scope='h3_lin'), drop_rate=drop_rate)

        return lrelu(h3), h3

    # filter num : design multi layer for 20, 256, 256
    def layer_v3(self, X, f_depth, f_height, f_width, f_filter, drop_rate):  # for size of 20, 256, 256
        conv_0 = conv3d(X, self.f_filter, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_0')
        maxpool_0 = maxpool3d(conv_0, k_d=1, k_h=2, k_w=2, s_d=1, s_h=2, s_w=2, name='max_3d_0')
        h0 = dropout(lrelu(self.bn0(maxpool_0)), drop_rate=drop_rate)

        conv_1 = conv3d(h0, self.f_filter * 2, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_1')
        maxpool_1 = maxpool3d(conv_1, k_d=1, k_h=2, k_w=2, s_d=1, s_h=2, s_w=2, name='max_3d_1')
        h1 = dropout(lrelu(self.bn1(maxpool_1)), drop_rate=drop_rate)

        conv_2 = conv3d(h1, self.f_filter * 4, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_2')
        h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)

        shape = h2.get_shape().as_list()
        h2 = tf.reshape(h2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        h3 = dropout(linear(h2, self.num_classes, scope='h3_lin'), drop_rate=drop_rate)

        return lrelu(h3), h3

    # filter num : 16,64,256
    def layer_v4(self, X, f_depth, f_height, f_width, f_filter, drop_rate):  # for size of 20, 256, 256
        conv_0 = conv3d(X, 16, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_0')
        maxpool_0 = maxpool3d(conv_0, k_d=1, k_h=2, k_w=2, s_d=1, s_h=2, s_w=2, name='max_3d_0')
        h0 = dropout(lrelu(self.bn0(maxpool_0)), drop_rate=drop_rate)

        conv_1 = conv3d(h0, 64, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_1')
        maxpool_1 = maxpool3d(conv_1, k_d=1, k_h=2, k_w=2, s_d=1, s_h=2, s_w=2, name='max_3d_1')
        h1 = dropout(lrelu(self.bn1(maxpool_1)), drop_rate=drop_rate)

        conv_2 = conv3d(h1, 256, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_2')
        h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)

        shape = h2.get_shape().as_list()
        h2 = tf.reshape(h2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        h3 = dropout(linear(h2, self.num_classes, scope='h3_lin'), drop_rate=drop_rate)

        return lrelu(h3), h3

    # PD Net version for Neurology
    def layer_v5(self, X, f_depth, f_height, f_width, f_filter, drop_rate):  # for size of 20, 256, 256, referred PD Net
        conv_0 = conv3d(X, 16, k_d=3, k_h=7, k_w=7, s_d=1, s_h=4, s_w=4, name='conv_3d_0')
        maxpool_0 = maxpool3d(conv_0, k_d=1, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='max_3d_0')
        h0 = dropout(lrelu(self.bn0(maxpool_0)), drop_rate=drop_rate)

        conv_1 = conv3d(h0, 64, k_d=3, k_h=5, k_w=5, s_d=1, s_h=1, s_w=1, name='conv_3d_1')
        maxpool_1 = maxpool3d(conv_1, k_d=1, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='max_3d_1')
        h1 = dropout(lrelu(self.bn1(maxpool_1)), drop_rate=drop_rate)

        conv_2 = conv3d(h1, 256, k_d=3, k_h=3, k_w=3, s_d=1, s_h=1, s_w=1, name='conv_3d_2')
        h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)

        shape = h2.get_shape().as_list()
        h2 = tf.reshape(h2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        h3 = dropout(linear(h2, self.num_classes, scope='h3_lin'), drop_rate=drop_rate)

        return lrelu(h3), h3

    # PD Net upgrade; add one layer at the last layer
    def layer_v6(self, X, f_depth, f_height, f_width, f_filter, drop_rate):  # for size of 20, 256, 256, referred PD Net
        conv_0 = conv3d(X, 16, k_d=3, k_h=7, k_w=7, s_d=1, s_h=4, s_w=4, name='conv_3d_0')
        maxpool_0 = maxpool3d(conv_0, k_d=1, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='max_3d_0')
        h0 = dropout(lrelu(self.bn0(maxpool_0)), drop_rate=drop_rate)

        conv_1 = conv3d(h0, 64, k_d=3, k_h=5, k_w=5, s_d=1, s_h=1, s_w=1, name='conv_3d_1')
        maxpool_1 = maxpool3d(conv_1, k_d=1, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='max_3d_1')
        h1 = dropout(lrelu(self.bn1(maxpool_1)), drop_rate=drop_rate)

        conv_2 = conv3d(h1, 256, k_d=3, k_h=3, k_w=3, s_d=1, s_h=1, s_w=1, name='conv_3d_2')
        h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)

        conv_3 = conv3d(h2, 256, k_d=3, k_h=3, k_w=3, s_d=1, s_h=1, s_w=1, padding='SAME',name='conv_3d_3')
        h3 = dropout(lrelu(self.bn3(conv_3)), drop_rate=drop_rate)

        shape = h3.get_shape().as_list()
        h3 = tf.reshape(h3, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        h4 = dropout(linear(h3, self.num_classes, scope='h4_lin'), drop_rate=drop_rate)

        return lrelu(h4), h4

    # PD Net upgrade; add one layer at the last layer
    def layer_v7_fail(self, X, f_depth, f_height, f_width, f_filter,
                 drop_rate):  # for size of 20, 256, 256, referred PD Net
        conv_0 = conv3d(X, 16, k_d=7, k_h=7, k_w=7, s_d=4, s_h=4, s_w=4, name='conv_3d_0')
        maxpool_0 = maxpool3d(conv_0, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='max_3d_0')
        h0 = dropout(lrelu(self.bn0(maxpool_0)), drop_rate=drop_rate)

        print("conv_0",conv_0.shape)

        conv_1 = conv3d(h0, 32, k_d=5, k_h=5, k_w=5, s_d=1, s_h=1, s_w=1, name='conv_3d_1')
        maxpool_1 = maxpool3d(conv_1, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='max_3d_1')
        h1 = dropout(lrelu(self.bn1(maxpool_1)), drop_rate=drop_rate)

        print("conv_1",conv_1.shape)

        conv_2 = conv3d(h1, 64, k_d=3, k_h=3, k_w=3, s_d=1, s_h=1, s_w=1, name='conv_3d_2')
        h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)

        print("conv_2",conv_2.shape)

        conv_3 = conv3d(h2, 256, k_d=3, k_h=3, k_w=3, s_d=1, s_h=1, s_w=1, padding='SAME', name='conv_3d_3')
        h3 = dropout(lrelu(self.bn3(conv_3)), drop_rate=drop_rate)

        print("conv_3",conv_3.shape)

        conv_4 = conv3d(h3, 512, k_d=3, k_h=3, k_w=3, s_d=1, s_h=1, s_w=1, padding='SAME', name='conv_3d_4')
        h4 = dropout(lrelu(self.bn4(conv_4)), drop_rate=drop_rate)

        print("conv_4",conv_4.shape)

        shape = h4.get_shape().as_list()
        h4 = tf.reshape(h4, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        h5 = dropout(lrelu(linear(h4, 2000, scope='h5_lin')), drop_rate=drop_rate)
        featurevs=h5
        print("h4", h4.shape)

        h6 = tf.reshape(h5, [-1, 2000])
        h6 = dropout(linear(h6, self.num_classes, scope='h6_lin'), drop_rate=drop_rate)

        print("h6",h6.shape)

        print('%s, shape = %s' % ('h5_lin', h5.get_shape()))

        return lrelu(h6), h6, featurevs

    # PD Net upgrade; add one layer at the last layer
    def layer_v7(self, X, f_depth, f_height, f_width, f_filter,drop_rate):  # for size of 20, 256, 256, referred PD Net
        conv_0 = conv3d(X, 16, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_0')
        #maxpool_0 = maxpool3d(conv_0, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='max_3d_0')
        h0 = dropout(lrelu(self.bn0(conv_0)), drop_rate=drop_rate)

        print("conv_0", conv_0.shape)

        conv_1 = conv3d(h0, 32, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_1')
        #maxpool_1 = maxpool3d(conv_1, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='max_3d_1')
        h1 = dropout(lrelu(self.bn1(conv_1)), drop_rate=drop_rate)

        print("conv_1", conv_1.shape)

        conv_2 = conv3d(h1, 64, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_2')
        h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)

        print("conv_2", conv_2.shape)

        conv_3 = conv3d(h2, 256, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, padding='SAME', name='conv_3d_3')
        h3 = dropout(lrelu(self.bn3(conv_3)), drop_rate=drop_rate)

        print("conv_3", conv_3.shape)

        conv_4 = conv3d(h3, 512, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, padding='SAME', name='conv_3d_4')
        h4 = dropout(lrelu(self.bn4(conv_4)), drop_rate=drop_rate)

        print("conv_4", conv_4.shape)

        shape = h4.get_shape().as_list()
        h4 = tf.reshape(h4, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        h5 = dropout(lrelu(linear(h4, 2000, scope='h5_lin')), drop_rate=drop_rate)

        print("h4", h4.shape)

        h6 = tf.reshape(h5, [-1, 2000])
        featurev = h6
        h7 = dropout(linear(h6, self.num_classes, scope='h6_lin'), drop_rate=drop_rate)

        print("h6", h7.shape)

        print('%s, shape = %s' % ('h7_lin', h7.get_shape()))

        return lrelu(h7), h7, featurev

    # PD Net upgrade; add one layer at the last layer

    # PD Net version, modified
    def PDNet_v2(self, X, f_depth, f_height, f_width, f_filter,
                 drop_rate):  # for size of 20, 256, 256, referred PD Net
        conv_0 = conv3d(X, 32, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_0')
        maxpool_0 = maxpool3d(conv_0, k_d=2, k_h=2, k_w=2, s_d=2, s_h=2, s_w=2, name='max_3d_0')
        h0 = dropout(lrelu(self.bn0(maxpool_0)), drop_rate=drop_rate)

        conv_1 = conv3d(h0, 64, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_1')
        maxpool_1 = maxpool3d(conv_1, k_d=2, k_h=2, k_w=2, s_d=2, s_h=2, s_w=2, name='max_3d_1')
        h1 = dropout(lrelu(self.bn1(maxpool_1)), drop_rate=drop_rate)

        conv_2 = conv3d(h1, 256, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_2')
        h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)

        shape = h2.get_shape().as_list()
        h2 = tf.reshape(h2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        h3 = dropout(linear(h2, self.num_classes, scope='h3_lin'), drop_rate=drop_rate)

        return lrelu(h3), h3

    # PD Net version, modified
    def PDNet_v0(self, X, f_depth, f_height, f_width, f_filter,
                     drop_rate):  # for size of 20, 256, 256, referred PD Net
        conv_0 = conv3d(X, 16, k_d=7, k_h=7, k_w=7, s_d=4, s_h=4, s_w=4, name='conv_3d_0')
        print("conv_0", conv_0)
        maxpool_0 = maxpool3d(conv_0, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='max_3d_0')
        print("max_pool_0", maxpool_0)
        h0 = dropout(lrelu(self.bn0(maxpool_0)), drop_rate=drop_rate)

        conv_1 = conv3d(h0, 64, k_d=5, k_h=5, k_w=5, s_d=1, s_h=1, s_w=1, name='conv_3d_1')
        print("conv_1", conv_1)
        maxpool_1 = maxpool3d(conv_1, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='max_3d_1')
        print("max_pool_1", maxpool_1)
        h1 = dropout(lrelu(self.bn1(maxpool_1)), drop_rate=drop_rate)

        conv_2 = conv3d(h1, 256, k_d=3, k_h=3, k_w=3, s_d=1, s_h=1, s_w=1, name='conv_3d_2')
        print("conv_2", conv_2)
        h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)

        shape = h2.get_shape().as_list()
        h2 = tf.reshape(h2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        print("h2", h2) # extracted feature from convolutional layer


        h3 = dropout(linear(h2, self.num_classes, scope='h3_lin'), drop_rate=drop_rate)

        return lrelu(h3), h3, h2

    def _conv3d(self, x, n_filters, kernels, strides, stddev=0.2, istrain=False,
                activation=None, bias=True, padding='SAME', name='conv3D'):
        """
        convolution 3D
        :param x:
        :param n_filters:
        :param kernels:
        :param strides:
        :param stddev:
        :param activation:
        :param bias:
        :param padding:
        :param name:
        :return:
        """
        assert len(kernels) == 3
        assert len(strides) == 3

        with tf.variable_scope(name):
            w = tf.get_variable('w',
                                [kernels[0], kernels[1], kernels[2], x.get_shape()[-1], n_filters],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv3d(x, w,
                                strides=[1, strides[0], strides[1], strides[2], 1],
                                padding=padding)
            if bias:
                b = tf.get_variable('b', [n_filters],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                conv = tf.nn.bias_add(conv, b)


            bn_name = "_bn"
            conv = tf.contrib.layers.batch_norm(conv,
                                         decay=self.bn_momentum,
                                         updates_collections=None,
                                         epsilon=self.bn_epsilon,
                                         scale=True,
                                         is_training=istrain,
                                         scope=name+bn_name)
            # if activation:
            #     conv = activation(conv)
            lrelu(conv, leak=0.1, name="lrelu")
            return conv

    def _spatial_reduction_block(self, net, name):
        """
        spatial reduction block used to reduct feature map size to half, meanwhile it will
        increase feature map size to twice.
        :param net:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            with tf.name_scope(name + '/Maxpool3d_2_2'):
               branch_0 = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1],
                                           strides=[1, 2, 2, 2, 1],
                                           padding='SAME')
            with tf.name_scope(name + '/Conv3d_a'):
                branch_1 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1] / 4, kernels=[3, 3, 3], istrain=not self.forward_only,
                                   strides=[2, 2, 2], activation=tf.nn.relu, name=name + '/b1_conv3_3', padding='SAME')
            with tf.name_scope(name + '/Conv3d_b'):
                branch_2 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1], kernels=[1, 1, 1], istrain=not self.forward_only,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b2_conv1_1', padding='SAME')
                branch_2 = self._conv3d(branch_2, n_filters=net.get_shape().as_list()[-1] * 5 / 16, kernels=[3, 3, 3], istrain=not self.forward_only,
                                   strides=[2, 2, 2], activation=tf.nn.relu, name=name + '/b2_conv3_3', padding='SAME')
            with tf.name_scope(name + '/Conv3d_c'):
                branch_3 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1], kernels=[1, 1, 1], istrain=not self.forward_only,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b3_conv1_1', padding='SAME')
                branch_3 = self._conv3d(branch_3, n_filters=net.get_shape().as_list()[-1] * 5 / 16, kernels=[3, 3, 3], istrain=not self.forward_only,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b3_conv3_3', padding='SAME')
                branch_3 = self._conv3d(branch_3, n_filters=net.get_shape().as_list()[-1] * 7 / 16, kernels=[3, 3, 3], istrain=not self.forward_only,
                                   strides=[2, 2, 2], activation=tf.nn.relu, name=name + '/b3_conv3_3_', padding='SAME')
            print('%s/branch_0, shape = %s' % (name, branch_0.get_shape()))
            print('%s/branch_1, shape = %s' % (name, branch_1.get_shape()))
            print('%s/branch_2, shape = %s' % (name, branch_2.get_shape()))
            print('%s/branch_3, shape = %s' % (name, branch_3.get_shape()))

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=4) # menas merging them as a tensor having lots of feature maps(fan out)

            return net

    def _residual_conv_block(self, net, name):
        """
        residual block, the number of feature map's size is unchange
        :param net:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            with tf.name_scope(name + '/Conv3d_a'):
                branch_0 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[3, 3, 3], istrain=not self.forward_only,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b0_conv3_3', padding='SAME')
            with tf.name_scope(name + '/Conv3d_b'):
                branch_1 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[1, 1, 1], istrain=not self.forward_only,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b1_conv1_1', padding='SAME')
                branch_1 = self._conv3d(branch_1, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[3, 3, 3], istrain=not self.forward_only,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b1_conv3_3', padding='SAME')
            with tf.name_scope(name + '/Conv3d_c'):
                branch_2 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[1, 1, 1], istrain=not self.forward_only,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b2_conv1_1', padding='SAME')
                branch_2 = self._conv3d(branch_2, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[3, 3, 3], istrain=not self.forward_only,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b2_conv3_3', padding='SAME')
                branch_2 = self._conv3d(branch_2, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[3, 3, 3], istrain=not self.forward_only,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b2_conv3_3_', padding='SAME')

            print('%s/branch_0, shape = %s' % (name, branch_0.get_shape()))
            print('%s/branch_1, shape = %s' % (name, branch_1.get_shape()))
            print('%s/branch_2, shape = %s' % (name, branch_2.get_shape()))

            with tf.name_scope(name + '/Merge'):
                concated = tf.concat(values=[branch_0, branch_1, branch_2], axis=4)
            with tf.name_scope(name + '/Conv3d_d'):
                concated = self._conv3d(concated, n_filters=net.get_shape().as_list()[-1], kernels=[1, 1, 1], istrain=not self.forward_only,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/concate_conv1_1',
                                   padding='SAME')
            with tf.name_scope(name + '/Residual_merge'):
                net = net + concated
                # net = tf.nn.relu(net)
                net = lrelu(net, leak=0.1, name="lrelu")
            return net


    #def residual_inception_c3d_net(self, x, dropout_prob, n_hidden_unit=2000, n_classes=2):
    def layer_v8(self, x, n_hidden_unit = 2000):
        n_hidden_unit = 512
        # n_hidden_unit :  num of node at last fc layer
        print('volume shape = %s' % x.get_shape())

        with tf.name_scope('reshape'):
            x = tf.reshape(x, [-1,
                               x.get_shape().as_list()[1],
                               x.get_shape().as_list()[2],
                               x.get_shape().as_list()[3],
                               1])

        print('after reshape, shape = %s' % x.get_shape())

        with tf.name_scope('conv1'):
            x = self._conv3d(x, n_filters=self.f_filter, kernels=[3, 3, 3], strides=[2, 2, 2], istrain=not self.forward_only,
                             activation=tf.nn.relu, name='conv1/conv3d', padding='SAME') # (?, D, H, W, f_filter)

        print('conv1, shape = %s' % x.get_shape())

        x = self._spatial_reduction_block(x, 'spatial_reduction_1')  # 18*18*18*128

        print('spatial_reduction_1, shape = %s' % x.get_shape())

        x = self._residual_conv_block(x, 'res_conv_block_1')  # 18*18*18*128

        print('res_conv_block_1, shape = %s' % x.get_shape())

        x = self._spatial_reduction_block(x, 'spatial_reduction_2')  # 9*9*9*256

        print('spatial_reduction_2, shape = %s' % x.get_shape())

        # x = self._spatial_reduction_block(x, 'spatial_reduction_3')  # 9*9*9*256
        #
        # print('spatial_reduction_3, shape = %s' % x.get_shape())

        x = self._residual_conv_block(x, 'res_conv_block_2')  # 9*9*9*256

        print('res_conv_block_2, shape = %s' % x.get_shape())


        with tf.name_scope('conv2'):
            x = self._conv3d(x, n_filters=x.get_shape().as_list()[-1] * 2, kernels=[3, 3, 3], strides=[2, 2, 2], istrain=not self.forward_only,
                        activation=tf.nn.relu, name='conv2/conv3d', padding='SAME')  # 9*9*9*128

        print('conv2, shape = %s' % x.get_shape())

        with tf.name_scope('maxpool1'):
            x = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')  # 4*4*4*128
        print('maxpool1, shape = %s' % x.get_shape())


        with tf.name_scope('conv3'):
            x = self._conv3d(x, n_filters=x.get_shape().as_list()[-1] * 2, kernels=[3, 3, 3], strides=[2, 2, 2], istrain=not self.forward_only,
                        activation=tf.nn.relu, name='conv3/conv3d', padding='SAME')  # 9*9*9*128
        print('conv3, shape = %s' % x.get_shape())

        # with tf.name_scope('maxpool2'):
        #     x = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')  # 4*4*4*128
        # print('maxpool2, shape = %s' % x.get_shape())

        with tf.name_scope('flatten'):
            x = tf.reshape(x, [-1, x.get_shape().as_list()[1] * x.get_shape().as_list()[2]
                               * x.get_shape().as_list()[3] * x.get_shape().as_list()[4]])
        print('flatten, shape = %s' % x.get_shape())

        with tf.name_scope('dropout1'):
            x = tf.nn.dropout(x, keep_prob=self.drop_rate)
        featurev = x

        with tf.name_scope('fc1'):
            #w1 = tf.get_variable("w1", shape=[x.get_shape().as_list()[-1], n_hidden_unit], initializer=tf.contrib.layers.xavier_initializer())
            #b1 = tf.get_variable("b1", shape=[n_hidden_unit], initializer=tf.contrib.layers.xavier_initializer())
            w1 = tf.Variable(tf.random_normal([x.get_shape().as_list()[-1], n_hidden_unit]))
            b1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_unit]))
            x = tf.matmul(x, w1) + b1

            # x = tf.contrib.layers.batch_norm(x,
            #                                     decay=self.bn_momentum,
            #                                     updates_collections=None,
            #                                     epsilon=self.bn_epsilon,
            #                                     scale=True,
            #                                     is_training=self.forward_only,
            #                                     scope='fc1' + "_bn")

            # x = tf.nn.relu(x)
            x=lrelu(x, leak=0.1, name="lrelu")
        print('fc1, shape = %s' % x.get_shape())

        with tf.name_scope('dropout2'):
            x = tf.nn.dropout(x, keep_prob=self.drop_rate)

        with tf.name_scope('fc2'):
            #w2 = tf.get_variable("w2", shape=[n_hidden_unit, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            #b2 = tf.get_variable("b2", shape=[self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.Variable(tf.random_normal([n_hidden_unit, self.num_classes]))
            b2 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))
            x = tf.matmul(x, w2) + b2

            # x = tf.contrib.layers.batch_norm(x,
            #                                  decay=self.bn_momentum,
            #                                  updates_collections=None,
            #                                  epsilon=self.bn_epsilon,
            #                                  scale=True,
            #                                  is_training=self.forward_only,
            #                                  scope='fc2' + "_bn")

            # x = tf.nn.relu(x)
            x = lrelu(x, leak=0.1, name="lrelu")
        print('fc2, shape = %s' % x.get_shape())
        return x, featurev

    def train(self, x, y):
        print("[*] Start Train Mode")

        writer = tf.summary.FileWriter('./tmp', graph=self.session.graph)

        if self.train_type == "tvt":
            print("tr_data", x[0].shape)
            print("val_data", x[1].shape)
            print("test_data", x[2].shape)
            print("tr_label", y[0].shape)
            print("val_label", y[1].shape)
            print("test_label", y[2].shape)
            train_batches_per_epoch = int(np.floor(len(x[0]) / self.batch_size))
            val_batches_per_epoch = int(np.floor(len(x[1]) / self.batch_size))
            test_batches_per_epoch = int(np.floor(len(x[2]) / self.batch_size))

        elif self.train_type == "tv":
            print("tr_data", x[0].shape)
            print("val_data", x[1].shape)
            print("tr_label", y[0].shape)
            print("val_label", y[1].shape)
            train_batches_per_epoch = int(np.floor(len(x[0]) / self.batch_size))
            val_batches_per_epoch = int(np.floor(len(x[1]) / self.batch_size))

        # mean subtraction
        mean = np.mean(x[0], axis=0).astype(np.int32)
        #mean = np.mean(x[0])
        print("train data mean", mean.shape)
        x[0] -= mean
        x[1] -= mean
        if self.train_type == "tvt":
            x[2] -= mean



        # apply fft to dataset
        # ftimage = np.fft.fftn(x[0])
        # print("DD##",np.array(ftimage).shape)
        # ftimage = np.fft.fftshift(ftimage)
        # x[0] = np.abs(ftimage)
        #
        # ftimage = np.fft.fftn(x[1])
        # ftimage = np.fft.fftshift(ftimage)
        # x[1] = np.abs(ftimage)
        # if self.train_type == "tvt":
        #     ftimage = np.fft.fftn(x[2])
        #     ftimage = np.fft.fftshift(ftimage)
        #     x[2] = np.abs(ftimage)



        ta = []
        va = []

        previous_losses = []
        current_step = 0
        step_print_step = 100
        steps_per_checkpoint = 200
        start_train_val_time = time.time()

        np.random.seed(0)
        for ep in range(self.num_epoch):

            total_train_acc = 0
            total_train_count = 0

            for ind in range(train_batches_per_epoch):

                start_time = time.time()
                len_tr_data = len(x[0])
                batch_mask = np.random.choice(len_tr_data, self.batch_size)
                batch_xs = x[0][batch_mask]
                batch_ys = y[0][batch_mask]

                step_time = 0.0
                # Data Augmentation
                # if (ep%2)==0:
                #     batch_xs = da_processor_3d(self.session, batch_xs)

                train_logits_, train_Yhat_, train_feature_vs_, _, loss, train_acc_= self.session.run([self.logits, self.Yhat, self.featurev, self.optimizer, self.loss, self.accuracy], feed_dict={self.X : batch_xs, self.Y : batch_ys, self.drop_rate:0.5})
                #writer.add_summary(summary, ep*train_batches_per_epoch+ind)

                total_train_acc += train_acc_
                total_train_count += 1
                step_time += (time.time() - start_time)/step_print_step
                current_step+=1
                if current_step % step_print_step == 0:
                    print("global step %d learning rate %.4f loss %f, step_time %.2f" % (current_step, self.learning_rate.eval(), loss, step_time))


                if self.lr_decay == "normal":
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        self.session.run(self.learning_rate_decay_op)
                        print("[!] loss is wooden, global step %d learning rate %f loss %f" % (current_step, self.learning_rate.eval(), loss))

                        if len(previous_losses)>10:
                            previous_losses=previous_losses[-3:]
                    previous_losses.append(loss)



            total_val_acc = 0
            total_val_count = 0
            v_conf_matrix = np.zeros([self.num_classes, self.num_classes])

            # ROC and extracting feature
            val_label = []
            val_y_score = []

            # feature analysis
            val_feature_vs = []

            for ind in range(val_batches_per_epoch):
                if ind==(val_batches_per_epoch-1):
                    batch_xs = x[1][ind*self.batch_size:]
                    batch_ys = y[1][ind*self.batch_size:]
                else :
                    batch_xs = x[1][ind*self.batch_size:(ind+1)*self.batch_size]
                    batch_ys = y[1][ind*self.batch_size:(ind+1)*self.batch_size]
                feature_vs_, val_acc_, logits_, yhat_ = self.session.run([self.featurev, self.accuracy, self.logits, self.Yhat], feed_dict={self.X:batch_xs, self.Y:batch_ys, self.drop_rate:1.0})
                total_val_acc+=val_acc_
                total_val_count+=1

                val_label.append(np.array(yhat_))
                val_y_score.append(np.array(logits_))
                val_feature_vs.append(feature_vs_)


                for idx in range(len(batch_xs)):
                    v_conf_matrix[np.argmax(yhat_[idx])][np.argmax(logits_[idx])]+=1


            train_acc = total_train_acc / total_train_count
            val_acc = total_val_acc/total_val_count
            print("epoch ", ep, ", Validation Acc", val_acc, ", Train Acc", train_acc)
            print("Confusion Matrix")
            print(v_conf_matrix)

            ta.append(train_acc)
            va.append(val_acc)


            #if current_step % steps_per_checkpoint == 0:
            #    print("global step %d learning rate %.4f loss %f" % (current_step, self.learning_rate.eval(), loss))
            #    self.saver.save(self.session, os.path.join(self.model_dir, self.checkpoint_name), global_step=current_step)

            print("Total train val time(min) : ", (time.time()-start_train_val_time)/60)



            if ep is self.num_epoch-1:
                # ROC and extracting feature
                train_label = []
                # feature analysis
                train_feature_vs = []

                # extract features for train set
                train_batches_per_epoch = int(np.floor(len(x[0]) / self.batch_size))
                for ind in range(train_batches_per_epoch):
                    if ind == (train_batches_per_epoch - 1):
                        batch_xs = x[0][ind * self.batch_size:]
                        batch_ys = y[0][ind * self.batch_size:]
                    else:
                        batch_xs = x[0][ind * self.batch_size:(ind + 1) * self.batch_size]
                        batch_ys = y[0][ind * self.batch_size:(ind + 1) * self.batch_size]
                    train_Yhat_, train_feature_vs_ = self.session.run([self.Yhat, self.featurev], feed_dict={self.X: batch_xs, self.Y: batch_ys, self.drop_rate: 1.0})

                    train_feature_vs.append(train_feature_vs_)
                    train_label.append(train_Yhat_)

                train_label = np.array(train_label)
                #train_label = train_label.reshape([train_label.shape[0] * train_label.shape[1], self.num_classes])
                train_label = np.concatenate(np.array(train_label), axis=0)


                val_label = np.array(val_label)
                val_label = val_label.reshape([val_label.shape[0] * val_label.shape[1], self.num_classes])
                val_y_score = np.array(val_y_score)
                val_y_score = val_y_score.reshape([val_y_score.shape[0] * val_y_score.shape[1], self.num_classes])

                train_feature_vs = np.array(train_feature_vs)
                #train_feature_vs = train_feature_vs.reshape([train_feature_vs.shape[0] * train_feature_vs.shape[1], train_feature_vs.shape[2]])
                train_feature_vs = np.concatenate(np.array(train_feature_vs), axis=0)
                print("train_feature",train_feature_vs.shape)
                # (N, M) need to be concatenated with label (N, num_classes), finally to be (N, M+num_classes)

                val_feature_vs = np.array(val_feature_vs)
                val_feature_vs = val_feature_vs.reshape([val_feature_vs.shape[0] * val_feature_vs.shape[1], val_feature_vs.shape[2]])
                # (N, M) need to be concatenated with label (N, num_classes), finally to be (N, M+num_classes)


                train_feature_vs = np.concatenate((train_feature_vs, train_label), axis=1)
                val_feature_vs = np.concatenate((val_feature_vs, val_label), axis=1)
                # print("#################3", feature_vs.shape)
                # print(,label.shape) # (60, 2)
                # print(y_score.shape) # (60, 2)

                # save some information to draw roc curve on validation/test
                draw_roc(val_y_score, val_label, 0, title="Net")
                concatlist = concatList_axis_1([val_label, val_y_score])
                save_list_info("ROC_delay2.csv", concatlist)

                # save feature information on train/validation/test phase
                save_list_info("train_feature_data.csv", train_feature_vs) #
                save_list_info("val_feature_data.csv", val_feature_vs)


        if self.train_type == "tvt":
            label = []
            y_score = []
            total_test_acc = 0
            total_test_count = 0

            conf_matrix = np.zeros([self.num_classes, self.num_classes])
            for ind in range(test_batches_per_epoch):
                if ind == (test_batches_per_epoch - 1):
                    batch_xs = x[2][ind * self.batch_size:]
                    batch_ys = y[2][ind * self.batch_size:]
                else:
                    batch_xs = x[2][ind * self.batch_size:(ind + 1) * self.batch_size]
                    batch_ys = y[2][ind * self.batch_size:(ind + 1) * self.batch_size]

                test_acc, logits_, yhat_ = self.session.run([self.accuracy,self.logits,self.Yhat], feed_dict={self.X:batch_xs, self.Y:batch_ys, self.drop_rate:1.0})
                total_test_acc += test_acc
                total_test_count += 1

                for idx in range(len(batch_xs)):
                    conf_matrix[np.argmax(yhat_[idx])][np.argmax(logits_[idx])]+=1


                label.append(yhat_)
                y_score.append(logits_)

                label = np.array(label)
                label = label.reshape([label.shape[0] * label.shape[1], self.num_classes])
                y_score = np.array(y_score)
                y_score = y_score.reshape([y_score.shape[0] * y_score.shape[1], self.num_classes])

                # Draw ROC curve
                draw_roc(y_score, label, 0, title="ROC")

            test_acc = total_test_acc / total_test_count
            print("Test Acc", test_acc)
            print("Confusion Matrix")
            print(conf_matrix)


        if self.visualize is True:
            import matplotlib.pyplot as plt
            import matplotlib
            # matplotlib.rcParams.update({'font.size':22})    #
            # matplotlib.rc('font', size=23)
            # matplotlib.rc('axes', labelsize=25)
            # matplotlib.rc('legend', fontsize=25)

            x = np.arange(len(ta))
            y1 = np.array(ta) * 100
            y2 = np.array(va) * 100

            plt.plot(x, y1, label="Train")
            plt.plot(x, y2, linestyle="--", label="Validation")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy(%)")
            plt.legend()
            plt.show()






    def predict(self):
        print("[*] Start Predict Mode")

        return


    def get_batch(self, datas): # num_classes, each_data_size, depth, height, width

        for ind in range(self.num_classes):
            random.shuffle(datas[ind])

        if self.train_type == "tvt":  # train() will treat test mode
            each_label = [[] for _ in range(3)]  # length of tr, val and test data

            tr_data=[]
            val_data=[]
            test_data=[]

            for ind, data in enumerate(datas):
                temp = data[:int(self.train_rate*len(data))]
                tr_data.append(temp[:int(0.9*len(temp))])
                val_data.append(temp[int(0.9*len(temp)):])
                test_data.append(data[int(self.train_rate*len(data)):])

                # each_label[0].append(len(temp[:int(0.9*len(temp))]))
                # each_label[1].append(len(temp[int(0.9*len(temp)):]))
                # each_label[2].append(len(data[int(self.train_rate*len(data)):]))

                each_label[0]+=[ind]*len(temp[:int(0.9*len(temp))]) # for tr
                each_label[1]+=[ind]*len(temp[int(0.9*len(temp)):]) # for val
                each_label[2]+=[ind]*len(data[int(self.train_rate*len(data)):]) # for test


            # return np.vstack(np.array(tr_data)), np.vstack(np.array(val_data)), np.vstack(np.array(test_data)),\
            #     np.array([0]*each_label[0][0]+[1]*each_label[0][1]), \
            #     np.array([0]*each_label[1][0]+[1]*each_label[1][1]), \
            #     np.array([0]*each_label[2][0]+[1]*each_label[2][1])

            return np.vstack(np.array(tr_data)), np.vstack(np.array(val_data)), np.vstack(np.array(test_data)),\
                    np.array(each_label[0]), np.array(each_label[1]), np.array(each_label[2])


        elif self.train_type == "tv":  # train() will treat only train and val
            each_label = [[] for _ in range(2)]  # length of tr and val data
            tr_data = []
            val_data = []
            for ind, data in enumerate(datas):
                tr_data.append(data[:int(self.train_rate*len(data))])
                val_data.append(data[int(self.train_rate*len(data)):])
                # each_label[0].append(len(data[:int(self.train_rate*len(data))]))
                # each_label[1].append(len(data[int(self.train_rate*len(data)):]))
                each_label[0] += [ind] * len(data[:int(self.train_rate*len(data))])  # for tr
                each_label[1] += [ind] * len(data[int(self.train_rate*len(data)):])  # for val

            return np.vstack(np.array(tr_data)), np.vstack(np.array(val_data)), \
                   np.array(each_label[0]), np.array(each_label[1])




























