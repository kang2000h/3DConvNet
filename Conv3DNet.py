import time
import os
import random
import math
import numpy as np
import tensorflow as tf

from utils import *
from exceptions import *
from dicom_loader import *
from sklearn.model_selection import StratifiedKFold

class Conv3DNet():
    def __init__(self, session, i_depth, i_height, i_width, i_cdim, num_classes, batch_size, num_epoch, model_dir, checkpoint_name,
                 lr_decay, visualize=False, learning_rate = 0.0001, learning_rate_decay_factor = 0.99, epochs_per_decay = 10, train_rate = 0.7, tv_type="holdout", f_d=5, f_h=5, f_w=5,
                 class_weights=None, f_filter=32, beta1=0.5, forward_only=False, transfer_learning=None):

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
        self.tv_type = tv_type
        self.class_weights  = class_weights
        self.f_filter = f_filter
        self.beta1 = beta1
        self.forward_only = forward_only
        self.is_transfer= transfer_learning

        self.visualize = visualize

        # Constants dictating the learning rate schedule.
        self.RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
        self.RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
        self.RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

        self.bn_epsilon = 1e-5
        self.bn_momentum = 0.9

        self.feature_list = []


    def create_model(self, transfer_learning=None, train_layers=None):

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
        #_, self.logits, self.featurev = self.PDNet_v2(self.X, self.i_depth, self.i_height, self.i_width, self.f_filter, self.drop_rate)
        _, self.logits, self.featurev = self.PDNet_v3(self.X, self.i_depth, self.i_height, self.i_width, self.f_filter,self.drop_rate)
        #_, self.logits, self.featurev  = self.layer_v7(self.X, self.i_depth, self.i_height, self.i_width, self.f_filter, self.drop_rate)
        #self.logits, self.featurev = self.layer_v8(self.X)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        # self.loss = tf.reduce_mean(
        #     sigmoid_cross_entropy_with_logits(x=self.logits, y=self.Yhat))


        self.loss = self.loss_function(batch_x = self.logits, batch_y = self.Yhat, class_weights=self.class_weights)
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


        # #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        # var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        # gradients = tf.gradients(self.loss, var_list)
        # gradients = list(zip(gradients, var_list))
        # self.optimizer = self.optimizer.apply_gradients(grads_and_vars=gradients)



    def load(self, model_dir):
        print("[*] Check model exist...")

        if os.path.isdir(self.model_dir) is False:
            print("[*] model_dir doesn't exist, create the directory... dir:", self.model_dir)
            os.makedirs(self.model_dir)

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path): # check whether created checkpoint_path is
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else :
            print("Create model with new parameters.")
            self.session.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()


    def loss_function(self, batch_x, batch_y=None, class_weights=None):
        class_weights=None
        if class_weights is None:
            flat_logits = tf.reshape(batch_x, [-1, self.num_classes])
            flat_labels = tf.reshape(batch_y, [-1, self.num_classes])
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.add_n([cross_entropy_mean] + regularization_losses)
            return self.loss
        else :
            flat_logits = tf.reshape(batch_x, [-1, self.num_classes])
            flat_labels = tf.reshape(batch_y, [-1, self.num_classes])

            class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

            weight_map = tf.multiply(flat_labels, class_weights)
            weight_map = tf.reduce_sum(weight_map, axis=1)

            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                               labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)

            cross_entropy_mean = tf.reduce_mean(weighted_loss)

            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.add_n([cross_entropy_mean] + regularization_losses)
            return self.loss


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

        print("h4", h4.shape)

        h6 = tf.reshape(h5, [-1, 2000])
        h6 = dropout(linear(h6, self.num_classes, scope='h6_lin'), drop_rate=drop_rate)

        print("h6",h6.shape)

        print('%s, shape = %s' % ('h5_lin', h5.get_shape()))

        return lrelu(h6), h6

    # PD Net upgrade; add one layer at the last layer
    def layer_v7(self, X, f_depth, f_height, f_width, f_filter,drop_rate):  # for size of 20, 256, 256, referred PD Net
        # n_hidden_unit :  num of node at last fc layer
        print('volume shape = %s' % X.get_shape())

        if self.forward_only is False and self.is_transfer is True:
            sometimes_trainable = False  # cnn layer
            always_trainable = True  # fc

        elif self.forward_only is False and self.is_transfer is False:
            sometimes_trainable = True
            always_trainable = True  # fc

        elif self.forward_only is True:
            sometimes_trainable = False
            always_trainable = False
        else:
            chk_integrity("chk this param")
            sometimes_trainable = True
            always_trainable = True

        # x = self._conv3d(X, n_filters=self.f_filter, kernels=[3, 3, 3], strides=[2, 2, 2], istrain=sometimes_trainable,
        #                  activation=tf.nn.relu, name='conv1/conv3d', padding='SAME')  # (?, D, H, W, f_filter)


        conv_0 = self._conv3d(X, n_filters=self.f_filter, kernels=[3,3,3], strides=[2,2,2], istrain=sometimes_trainable,
                              activation=tf.nn.relu, name='conv_0', padding='SAME')

        maxpool_0 = maxpool3d(conv_0, k_d=2, k_h=2, k_w=2, s_d=2, s_h=2, s_w=2, name='max_3d_0')
        h0 = tf.nn.dropout(maxpool_0, keep_prob=self.drop_rate)
        #h0 = dropout(lrelu(self.bn0(conv_0)), drop_rate=drop_rate)

        print("conv_0", conv_0.shape)

        conv_1 = self._conv3d(h0, n_filters=self.f_filter*2, kernels=[3, 3, 3], strides=[2, 2, 2],
                              istrain=sometimes_trainable,
                              activation=tf.nn.relu, name='conv_1', padding='SAME')

        #maxpool_1 = maxpool3d(conv_1, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='max_3d_1')
        #h1 = dropout(lrelu(self.bn1(conv_1)), drop_rate=drop_rate)
        h1 = tf.nn.dropout(conv_1, keep_prob=self.drop_rate)

        print("conv_1", conv_1.shape)

        conv_2 = self._conv3d(h1, n_filters=self.f_filter * 4, kernels=[3, 3, 3], strides=[2, 2, 2],
                              istrain=sometimes_trainable,
                              activation=tf.nn.relu, name='conv_2', padding='SAME')

        #conv_2 = conv3d(h1, 64, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_2')
        # h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)
        h2 = tf.nn.dropout(conv_2, keep_prob=self.drop_rate)

        print("conv_2", conv_2.shape)

        conv_3 = self._conv3d(h2, n_filters=self.f_filter * 8, kernels=[3, 3, 3], strides=[2, 2, 2],
                              istrain=sometimes_trainable,
                              activation=tf.nn.relu, name='conv_3', padding='SAME')
        h3 = tf.nn.dropout(conv_3, keep_prob=self.drop_rate)
        # conv_3 = conv3d(h2, 256, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, padding='SAME', name='conv_3d_3')
        # h3 = dropout(lrelu(self.bn3(conv_3)), drop_rate=drop_rate)

        print("conv_3", conv_3.shape)

        conv_4 = self._conv3d(h3, n_filters=self.f_filter * 16, kernels=[3, 3, 3], strides=[2, 2, 2],
                              istrain=sometimes_trainable,
                              activation=tf.nn.relu, name='conv_4', padding='SAME')
        h4 = tf.nn.dropout(conv_4, keep_prob=self.drop_rate)

        #conv_4 = conv3d(h3, 512, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, padding='SAME', name='conv_3d_4')
        # h4 = dropout(lrelu(self.bn4(conv_4)), drop_rate=drop_rate)

        print("conv_4", conv_4.shape)

        conv_5 = self._conv3d(h4, n_filters=self.f_filter * 16, kernels=[3, 3, 3], strides=[2, 2, 2],
                              istrain=sometimes_trainable,
                              activation=tf.nn.relu, name='conv_5', padding='SAME')
        h5 = tf.nn.dropout(conv_5, keep_prob=self.drop_rate)

        print("conv_5", conv_5.shape)

        shape = h5.get_shape().as_list()
        h5 = tf.reshape(h5, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        print("h5", h5.shape)
        featurev = h5
        h6 = dropout(lrelu(linear(h5, 172, scope='h5_lin')), drop_rate=drop_rate)
        print("h6", h6.shape)
        h6 = tf.reshape(h6, [-1, 172])

        h7 = dropout(linear(h6, 116, scope='h6_lin'), drop_rate=drop_rate)
        print("h7", h7.shape)
        h7 = tf.reshape(h7, [-1, 116])

        h8 = dropout(lrelu(linear(h7, self.num_classes, scope='h7_lin')), drop_rate=drop_rate)
        print("h8", h8.shape)
        h8 = tf.reshape(h8, [-1, self.num_classes])


        print('%s, shape = %s' % ('h8_lin', h8.get_shape()))

        return lrelu(h8), h8, featurev


    # PD Net version, modified only filter size
    def PDNet_v2(self, X, f_depth, f_height, f_width, f_filter,
                 drop_rate):  # for size of 20, 256, 256, referred PD Net
        conv_0 = conv3d(X, 32, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_0')
        print("conv_0", conv_0)

        maxpool_0 = maxpool3d(conv_0, k_d=2, k_h=2, k_w=2, s_d=2, s_h=2, s_w=2, name='max_3d_0')
        h0 = dropout(lrelu(self.bn0(maxpool_0)), drop_rate=drop_rate)

        conv_1 = conv3d(h0, 64, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_1')
        print("conv_1", conv_1)

        maxpool_1 = maxpool3d(conv_1, k_d=2, k_h=2, k_w=2, s_d=2, s_h=2, s_w=2, name='max_3d_1')
        h1 = dropout(lrelu(self.bn1(maxpool_1)), drop_rate=drop_rate)

        conv_2 = conv3d(h1, 256, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_2')
        print("conv_2",conv_2)


        h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)

        shape = h2.get_shape().as_list()
        h2 = tf.reshape(h2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        print("h2", h2)
        h3 = dropout(linear(h2, self.num_classes, scope='h3_lin'), drop_rate=drop_rate)

        return lrelu(h3), h3, h2

    # PD Net version, modified filter size, additional MLP
    def PDNet_v3(self, X, f_depth, f_height, f_width, f_filter, drop_rate):  # for size of 20, 256, 256, referred PD Net
        conv_0 = conv3d(X, self.f_filter, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_0')
        print("conv_0", conv_0)
        self.feature_list.append(conv_0)
        maxpool_0 = maxpool3d(conv_0, k_d=2, k_h=2, k_w=2, s_d=2, s_h=2, s_w=2, name='max_3d_0')
        h0 = dropout(lrelu(self.bn0(maxpool_0)), drop_rate=drop_rate)

        conv_1 = conv3d(h0, self.f_filter*2, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_1')
        print("conv_1", conv_1)
        self.feature_list.append(conv_1)
        maxpool_1 = maxpool3d(conv_1, k_d=2, k_h=2, k_w=2, s_d=2, s_h=2, s_w=2, name='max_3d_1')
        h1 = dropout(lrelu(self.bn1(maxpool_1)), drop_rate=drop_rate)

        conv_2 = conv3d(h1, self.f_filter*4, k_d=3, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name='conv_3d_2')
        print("conv_2", conv_2)
        self.feature_list.append(conv_2)

        h2 = dropout(lrelu(self.bn2(conv_2)), drop_rate=drop_rate)

        shape = h2.get_shape().as_list()
        h2 = tf.reshape(h2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        print("h2", h2)

        h3 = dropout(linear(h2, 1024, scope='h3_lin'), drop_rate=drop_rate)
        h3 = dropout(linear(h3, 512, scope='h4_lin'), drop_rate=drop_rate)
        h3 = dropout(linear(h3, 512, scope='h5_lin'), drop_rate=drop_rate)

        h3 = dropout(linear(h3, self.num_classes, scope='h6_lin'), drop_rate=drop_rate)
        return lrelu(h3), h3, h2

    # PD Net version, modified with additional MLP
    def PDNet_v1(self, X, f_depth, f_height, f_width, f_filter,
                     drop_rate):  # for size of 20, 256, 256, referred PD Net
        conv_0 = conv3d(X, 16, k_d=7, k_h=7, k_w=7, s_d=4, s_h=4, s_w=4,  name='conv_3d_0')
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


        h3 = dropout(linear(h2, 1024, scope='h3_lin'), drop_rate=drop_rate)
        h3 = dropout(linear(h3, 512, scope='h4_lin'), drop_rate=drop_rate)
        h3 = dropout(linear(h3, 512, scope='h5_lin'), drop_rate=drop_rate)

        h3 = dropout(linear(h3, self.num_classes, scope='h7_lin'), drop_rate=drop_rate)

        return lrelu(h3), h3, h2

    # PD Net version, original
    def PDNet_v0(self, X, f_depth, f_height, f_width, f_filter, drop_rate):  # for size of 20, 256, 256, referred PD Net
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
        print("h2", h2)  # extracted feature from convolutional layer
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
                                initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=istrain)
            conv = tf.nn.conv3d(x, w,
                                strides=[1, strides[0], strides[1], strides[2], 1],
                                padding=padding)
            if bias:
                b = tf.get_variable('b', [n_filters],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=istrain)
                conv = tf.nn.bias_add(conv, b)


            bn_name = "_bn"
            conv = tf.contrib.layers.batch_norm(conv,
                                         decay=self.bn_momentum,
                                         updates_collections=None,
                                         epsilon=self.bn_epsilon,
                                         scale=True,
                                         is_training=istrain,
                                         scope=name+bn_name)
            if activation:
                #conv = activation(conv)
                conv = lrelu(conv, leak=0.2, name="lrelu")
            return conv

    def _spatial_reduction_block(self, net, name, istrain=None):
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
                branch_1 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1] / 4, kernels=[3, 3, 3], istrain=istrain,
                                   strides=[2, 2, 2], activation=tf.nn.relu, name=name + '/b1_conv3_3', padding='SAME')
            with tf.name_scope(name + '/Conv3d_b'):
                branch_2 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1], kernels=[1, 1, 1], istrain=istrain,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b2_conv1_1', padding='SAME')
                branch_2 = self._conv3d(branch_2, n_filters=net.get_shape().as_list()[-1] * 5 / 16, kernels=[3, 3, 3], istrain=istrain,
                                   strides=[2, 2, 2], activation=tf.nn.relu, name=name + '/b2_conv3_3', padding='SAME')
            with tf.name_scope(name + '/Conv3d_c'):
                branch_3 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1], kernels=[1, 1, 1], istrain=istrain,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b3_conv1_1', padding='SAME')
                branch_3 = self._conv3d(branch_3, n_filters=net.get_shape().as_list()[-1] * 5 / 16, kernels=[3, 3, 3], istrain=istrain,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b3_conv3_3', padding='SAME')
                branch_3 = self._conv3d(branch_3, n_filters=net.get_shape().as_list()[-1] * 7 / 16, kernels=[3, 3, 3], istrain=istrain,
                                   strides=[2, 2, 2], activation=tf.nn.relu, name=name + '/b3_conv3_3_', padding='SAME')
            print('%s/branch_0, shape = %s' % (name, branch_0.get_shape()))
            print('%s/branch_1, shape = %s' % (name, branch_1.get_shape()))
            print('%s/branch_2, shape = %s' % (name, branch_2.get_shape()))
            print('%s/branch_3, shape = %s' % (name, branch_3.get_shape()))

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=4) # menas merging them as a tensor having lots of feature maps(fan out)

            return net

    def _residual_conv_block(self, net, name, istrain=None):
        """
        residual block, the number of feature map's size is unchange
        :param net:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            with tf.name_scope(name + '/Conv3d_a'):
                branch_0 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[3, 3, 3], istrain=istrain,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b0_conv3_3', padding='SAME')
            with tf.name_scope(name + '/Conv3d_b'):
                branch_1 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[1, 1, 1], istrain=istrain,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b1_conv1_1', padding='SAME')
                branch_1 = self._conv3d(branch_1, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[3, 3, 3], istrain=istrain,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b1_conv3_3', padding='SAME')
            with tf.name_scope(name + '/Conv3d_c'):
                branch_2 = self._conv3d(net, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[1, 1, 1], istrain=istrain,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b2_conv1_1', padding='SAME')
                branch_2 = self._conv3d(branch_2, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[3, 3, 3], istrain=istrain,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b2_conv3_3', padding='SAME')
                branch_2 = self._conv3d(branch_2, n_filters=net.get_shape().as_list()[-1] / 2, kernels=[3, 3, 3], istrain=istrain,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/b2_conv3_3_', padding='SAME')

            print('%s/branch_0, shape = %s' % (name, branch_0.get_shape()))
            print('%s/branch_1, shape = %s' % (name, branch_1.get_shape()))
            print('%s/branch_2, shape = %s' % (name, branch_2.get_shape()))

            with tf.name_scope(name + '/Merge'):
                concated = tf.concat(values=[branch_0, branch_1, branch_2], axis=4)
            with tf.name_scope(name + '/Conv3d_d'):
                concated = self._conv3d(concated, n_filters=net.get_shape().as_list()[-1], kernels=[1, 1, 1], istrain=istrain,
                                   strides=[1, 1, 1], activation=tf.nn.relu, name=name + '/concate_conv1_1',
                                   padding='SAME')
            with tf.name_scope(name + '/Residual_merge'):
                net = net + concated

                #net = tf.nn.relu(net)
                net = lrelu(net, leak=0.2, name="lrelu")
            return net


    #def residual_inception_c3d_net(self, x, dropout_prob, n_hidden_unit=2000, n_classes=2):
    def layer_v8(self, x, n_hidden_unit = 2000):
        n_hidden_unit = 512
        # n_hidden_unit :  num of node at last fc layer
        print('volume shape = %s' % x.get_shape())


        if self.forward_only is False and self.is_transfer is True:
            sometimes_trainable=False # cnn layer
            always_trainable=True # fc

        elif self.forward_only is False and self.is_transfer is False:
            sometimes_trainable=True
            always_trainable=True # fc

        elif self.forward_only is True:
            sometimes_trainable=False
            always_trainable= False
        else :
            chk_integrity("chk this param")
            sometimes_trainable = True
            always_trainable = True




        with tf.name_scope('reshape'):
            x = tf.reshape(x, [-1,
                               x.get_shape().as_list()[1],
                               x.get_shape().as_list()[2],
                               x.get_shape().as_list()[3],
                               1])

        print('after reshape, shape = %s' % x.get_shape())

        with tf.name_scope('conv1'):
            x = self._conv3d(x, n_filters=self.f_filter, kernels=[1, 1, 1], strides=[1, 1, 1],  istrain=sometimes_trainable,
                             activation=tf.nn.relu, name='conv1/conv3d', padding='SAME') # (?, D, H, W, f_filter)

        print('conv1, shape = %s' % x.get_shape())

        x = self._spatial_reduction_block(x, 'spatial_reduction_1', sometimes_trainable)  # 18*18*18*128

        print('spatial_reduction_1, shape = %s' % x.get_shape())

        x = self._residual_conv_block(x, 'res_conv_block_1', sometimes_trainable)  # 18*18*18*128

        print('res_conv_block_1, shape = %s' % x.get_shape())

        x = self._spatial_reduction_block(x, 'spatial_reduction_2', sometimes_trainable)  # 9*9*9*256

        print('spatial_reduction_2, shape = %s' % x.get_shape())

        #x = self._spatial_reduction_block(x, 'spatial_reduction_3')  # 9*9*9*256

        #print('spatial_reduction_3, shape = %s' % x.get_shape())

        x = self._residual_conv_block(x, 'res_conv_block_2',sometimes_trainable)  # 9*9*9*256
        #
        print('res_conv_block_2, shape = %s' % x.get_shape())


        with tf.name_scope('conv2'):
            x = self._conv3d(x, n_filters=x.get_shape().as_list()[-1] * 2, kernels=[3, 3, 3], strides=[1, 1, 1], istrain=sometimes_trainable,
                        activation=tf.nn.relu, name='conv2/conv3d', padding='SAME')  # 9*9*9*128

        print('conv2, shape = %s' % x.get_shape())

        with tf.name_scope('maxpool1'):
            x = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')  # 4*4*4*128
        print('maxpool1, shape = %s' % x.get_shape())


        with tf.name_scope('conv3'):
            x = self._conv3d(x, n_filters=x.get_shape().as_list()[-1] * 2, kernels=[3, 3, 3], strides=[1, 1, 1], istrain=sometimes_trainable,
                        activation=tf.nn.relu, name='conv3/conv3d', padding='SAME')  # 9*9*9*128
        print('conv3, shape = %s' % x.get_shape())

        with tf.name_scope('maxpool2'):
            x = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')  # 4*4*4*128
        print('maxpool2, shape = %s' % x.get_shape())


        with tf.name_scope('flatten'):
            x = tf.reshape(x, [-1, x.get_shape().as_list()[1] * x.get_shape().as_list()[2]
                               * x.get_shape().as_list()[3] * x.get_shape().as_list()[4]])
        print('flatten, shape = %s' % x.get_shape())

        with tf.name_scope('dropout1'):
            x = tf.nn.dropout(x, keep_prob=self.drop_rate)
        featurev = x
        print("feature", x)

        with tf.name_scope('fc1'):
            #w1 = tf.get_variable("w1", shape=[x.get_shape().as_list()[-1], n_hidden_unit], initializer=tf.contrib.layers.xavier_initializer())
            #b1 = tf.get_variable("b1", shape=[n_hidden_unit], initializer=tf.contrib.layers.xavier_initializer())
            w1 = tf.Variable(tf.random_normal([x.get_shape().as_list()[-1], n_hidden_unit]), trainable=always_trainable)
            b1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_unit]), trainable=always_trainable)
            x = tf.matmul(x, w1) + b1

            # x = tf.contrib.layers.batch_norm(x,
            #                                     decay=self.bn_momentum,
            #                                     updates_collections=None,
            #                                     epsilon=self.bn_epsilon,
            #                                     scale=True,
            #                                     is_training=always_trainable,
            #                                     scope='fc1' + "_bn")

            #x = tf.nn.relu(x)
            x = lrelu(x, leak=0.2)
        print('fc1, shape = %s' % x.get_shape())

        with tf.name_scope('dropout2'):
            x = tf.nn.dropout(x, keep_prob=self.drop_rate)

        with tf.name_scope('fc2'):
            #w2 = tf.get_variable("w2", shape=[n_hidden_unit, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            #b2 = tf.get_variable("b2", shape=[self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.Variable(tf.random_normal([n_hidden_unit, n_hidden_unit]), trainable=always_trainable)
            b2 = tf.Variable(tf.constant(0.1, shape=[n_hidden_unit]), trainable=always_trainable)
            x = tf.matmul(x, w2) + b2

            # x = tf.contrib.layers.batch_norm(x,
            #                                  decay=self.bn_momentum,
            #                                  updates_collections=None,
            #                                  epsilon=self.bn_epsilon,
            #                                  scale=True,
            #                                  is_training=always_trainable,
            #                                  scope='fc2' + "_bn")

            #x = tf.nn.relu(x)
            x = lrelu(x, leak=0.2)
        print('fc2, shape = %s' % x.get_shape())

        with tf.name_scope('fc3'):
            #w2 = tf.get_variable("w2", shape=[n_hidden_unit, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            #b2 = tf.get_variable("b2", shape=[self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.Variable(tf.random_normal([n_hidden_unit, self.num_classes]), trainable=always_trainable)
            b2 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), trainable=always_trainable)
            x = tf.matmul(x, w2) + b2
            # x = tf.nn.relu(x)
            x = lrelu(x, leak=0.2)

            x = tf.contrib.layers.batch_norm(x,
                                             decay=self.bn_momentum,
                                             updates_collections=None,
                                             epsilon=self.bn_epsilon,
                                             scale=True,
                                             is_training=always_trainable,
                                             scope='fc3' + "_bn")
            # x = tf.nn.relu(x)
            x = lrelu(x, leak=0.2)
        print('fc3, shape = %s' % x.get_shape())
        return x, featurev

    # on validation/testing phase
    # need to return the results of prediction; acc
    # plot confusion matrix
    def predict(self, x, y, phase_name=None, roc_ft_extract=False, is_loocv=False):
        print("[*] Start Predict Mode")
        self.forward_only = True

        print("X to predict", x.shape, "on ", phase_name)
        print("label to predict", y.shape, "on ", phase_name)

        # mean subtraction
        x -= self.mean

        if is_loocv is True:
            print("[*] val on loocv ")
            temp = self.batch_size
            self.batch_size=1
            batch_xs = x
            batch_ys = y

            acc_ = self.session.run(
                [self.accuracy],
                feed_dict={self.X: batch_xs, self.Y: batch_ys, self.drop_rate: 1.0})

            self.batch_size = temp
            print("loocv decision", acc_)
            return [acc_]*4

        val_batches_per_epoch = int(np.floor(len(x) / self.batch_size))

        total_val_acc = 0
        total_val_count = 0
        v_conf_matrix = np.zeros([self.num_classes, self.num_classes])

        # ROC and extracting feature
        val_label = [] # one-hot encoding
        val_y_score = [] # one-hot encoding
        vacc = []
        # feature analysis
        val_feature_vs = []





        for ind in range(val_batches_per_epoch):

            if ind == (val_batches_per_epoch - 1):
                batch_xs = x[ind * self.batch_size:]
                batch_ys = y[ind * self.batch_size:]
            else:
                batch_xs = x[ind * self.batch_size:(ind + 1) * self.batch_size]
                batch_ys = y[ind * self.batch_size:(ind + 1) * self.batch_size]


            feature_vs_, val_acc_, logits_, yhat_ = self.session.run(
                [self.featurev, self.accuracy, self.logits, self.Yhat],
                feed_dict={self.X: batch_xs, self.Y: batch_ys, self.drop_rate: 1.0})

            total_val_acc += val_acc_
            total_val_count += 1

            val_label.append(yhat_)
            val_y_score.append(logits_)
            val_feature_vs.append(feature_vs_)


            for idx in range(len(batch_xs)):
                v_conf_matrix[np.argmax(yhat_[idx])][np.argmax(logits_[idx])] += 1


        val_acc = total_val_acc / total_val_count
        print(phase_name, " Acc : ", val_acc)
        print("Confusion Matrix")
        print(v_conf_matrix)
        with open("./results/"+phase_name+"_conf_matrix.csv", "w") as f:
            for row in v_conf_matrix:
                f.write("%s\n"%row)

        vacc.append(val_acc)
        print("\n\n")

        val_feature_vs = np.concatenate(val_feature_vs)
        val_label = np.concatenate(val_label)
        val_y_score = np.concatenate(val_y_score)

        if roc_ft_extract is True:

            val_feature_vs_and_l = concatList_axis_1([val_feature_vs, val_label])
            save_list_info("./results/" + phase_name + "_val_feature_data.csv", val_feature_vs_and_l)
            plot_roc_curve(val_label, val_y_score, label=None, filepath="./results/ROC_curve_"+phase_name+".png")

        return val_acc, val_label, val_y_score, val_feature_vs

    def train_v2(self, x, y, val_x=None, val_y=None, opserve_v=None, save_mode=None, phase_name=None, loss_threshold=None):

        print("[*] Start train v2 phase...")
        self.forward_only = False

        if opserve_v is None:
            print("You need to determine whether to split input data into 2 parts as train/val set")
            raise chk_integrity()
        elif save_mode is None:
            print("You need to determine whether to save model, if you want to do CV then need to unset the flag for save_mode")
            raise chk_integrity()

        train_data = np.array(x)
        train_label = np.array(y)
        print("train_data", train_data.shape)
        print("train_label", train_label.shape)
        # mean subtraction
        # self.mean = np.mean(train_data, axis=0).astype(np.int32)
        self.mean = train_data.mean(axis=0).astype(np.int32)
        #self.mean=0
        # self.mean = train_data.mean().astype(np.int32)
        # mean = np.mean(x[0])
        try:
            print("train data mean", self.mean.shape)
        except AttributeError as atbe:
            pass
        train_data -= self.mean

        if opserve_v is True:
            val_data = val_x
            val_label = val_y
            print("val_data", val_data.shape)
            print("val_label", val_label.shape)
            val_data -= self.mean


        train_batches_per_epoch = int(np.floor(len(x) / self.batch_size))

        ta_list = []
        va_list = []

        previous_losses = []
        current_step = 0
        step_print_step = 100
        #steps_per_checkpoint = 100
        start_train_val_time = time.time()


        for ep in range(self.num_epoch):

            total_train_acc = 0
            total_train_count = 0
            np.random.seed(int(round(time.time())) % 1000)
            for ind in range(train_batches_per_epoch):
                if ind == (train_batches_per_epoch - 1):
                    batch_xs = x[ind * self.batch_size:]
                    batch_ys = y[ind * self.batch_size:]
                else:
                    batch_xs = x[ind * self.batch_size:(ind + 1) * self.batch_size]
                    batch_ys = y[ind * self.batch_size:(ind + 1) * self.batch_size]
                start_time = time.time()
                # len_tr_data = len(train_data)
                # batch_mask = np.random.choice(len_tr_data, self.batch_size)
                # batch_xs = train_data[batch_mask]
                # batch_ys = train_label[batch_mask]

                step_time = 0.0
                # if (ep%2)==0:
                #     batch_xs = da_processor_3d(self.session, batch_xs)

                train_logits_, train_Yhat_, train_feature_vs_, _, loss, train_acc_ = self.session.run(
                    [self.logits, self.Yhat, self.featurev, self.optimizer, self.loss, self.accuracy],
                    feed_dict={self.X: batch_xs, self.Y: batch_ys, self.drop_rate: 0.5})
                # writer.add_summary(summary, ep*train_batches_per_epoch+ind)

                total_train_acc += train_acc_
                total_train_count += 1
                step_time += (time.time() - start_time) / step_print_step
                current_step += 1
                if current_step % step_print_step == 0:
                    print("global step %d learning rate %.4f loss %f, step_time %.2f" % (
                    current_step, self.learning_rate.eval(), loss, step_time))

                if self.lr_decay == "normal":
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        self.session.run(self.learning_rate_decay_op)
                        print("[!] loss is wooden, global step %d learning rate %f loss %f" % (
                        current_step, self.learning_rate.eval(), loss))

                        if len(previous_losses) > 10:
                            previous_losses = previous_losses[-3:]
                    previous_losses.append(loss)

                    if loss < loss_threshold:
                        break


            train_acc = total_train_acc / total_train_count
            ta_list.append(train_acc)
            print("[*]", ep, "epoch, Train Acc : ", train_acc, "loss : ", loss)

            if save_mode is True:
                print("[!] Save Model Parameter")
                #if current_step % steps_per_checkpoint == 0:
                #     print("global step %d learning rate %.4f loss %f" % (current_step, self.learning_rate.eval(), loss))
                self.saver.save(self.session, os.path.join(self.model_dir, self.checkpoint_name), global_step=current_step)

            val_y_score = None
            val_feature_vs = None

            if opserve_v is True:
                print("[*] Predict validation set...")
                val_acc, val_y_label, val_y_score, val_feature_vs = self.predict(val_x, val_y, phase_name="Validation", roc_ft_extract=False)
                # return val_acc, np.array(val_label), np.array(val_y_score), np.array(val_feature_vs)
                va_list.append(val_acc)

            print("Total train val time(min) : ", (time.time() - start_train_val_time) / 60)

            # ROC and extracting feature

            # feature analysis
            train_feature_vs = []
            # ROC, feature extraction

            if ep is self.num_epoch - 1 or (loss_threshold is not None and loss_threshold > loss):
                if (loss_threshold is not None and loss_threshold > loss):
                    print("loss is less than loss_threshold, loss : ", loss, "loss_threshold : ", loss_threshold)


                print("[*] Obtaining Train Features...")
                # extract features for train set
                _, train_label, _, train_feature_vs = self.predict(x, y, phase_name="Train Features", roc_ft_extract=False)

                #print("DDDDDDDDd, train_label", np.array(train_label).shape) # (N, num_classes)

                #train_label = train_label.reshape([train_label.shape[0] * train_label.shape[1], self.num_classes])
                #val_label = val_label.reshape([val_label.shape[0] * val_label.shape[1], self.num_classes])
                #val_y_score = val_y_score.reshape([val_y_score.shape[0] * val_y_score.shape[1], self.num_classes])

                #train_feature_vs = np.array(train_feature_vs)
                # train_feature_vs = train_feature_vs.reshape([train_feature_vs.shape[0] * train_feature_vs.shape[1], train_feature_vs.shape[2]])
                #train_feature_vs = np.concatenate(np.array(train_feature_vs), axis=0)

                # (N, M) need to be concatenated with label (N, num_classes), finally to be (N, M+num_classes)


                #val_feature_vs = val_feature_vs.reshape([val_feature_vs.shape[0] * val_feature_vs.shape[1], val_feature_vs.shape[2]])
                # (N, M) need to be concatenated with label (N, num_classes), finally to be (N, M+num_classes)

                # print("DD train_feature_vs", np.array(train_feature_vs).shape)
                # print("DD train_label", np.array(train_label).shape)

                #train_feature_vs = np.concatenate((train_feature_vs, train_label), axis=1)
                concatlist = concatList_axis_1([train_feature_vs, train_label])
                save_list_info("./results/" + phase_name + "_train_feature_data.csv", concatlist)

                # print("#################3", feature_vs.shape)
                # print(,label.shape) # (60, 2)
                # print(y_score.shape) # (60, 2)

                # save some information to draw roc curve on validation/test

                #draw_roc(val_y_score, val_label, 0, title="Net")
                if opserve_v is True:
                    #plot_roc_curve(val_y_label, val_y_score, label=None, filepath="./results/ROC_curve_"+phase_name)

                    val_feature_vs = np.array(val_feature_vs)
                    val_feature_vs = np.concatenate((val_feature_vs, val_y_label), axis=1)

                    concatlist = concatList_axis_1([val_y_label, val_y_score])
                    save_list_info("./results/ROC_data_"+phase_name+".csv", concatlist)

                    # save feature information on train/validation/test phase
                    save_list_info("./results/val_feature_data_"+phase_name+".csv", val_feature_vs)


                    if self.visualize is True :
                        plot_tv_trend("./results/tv_trend_"+phase_name+".png", ta_list, va_list)
                break
        return np.array(train_feature_vs)


    def get_batch_v2(self, datas, mode=None, ratio=None, islabel=None): # num_classes, each_data_size, depth, height, width

        if islabel is None:
            chk_integrity("check islabel argument, it is expected to be non None")
        num_classes = len(datas)
        X = []
        label = []
        voxels_shape = np.array(datas[0][0]).shape # hypothesis : datas have 3 or more dimension
        for ind, each_class in enumerate(datas):
            for each_c_voxels in each_class:
                X.append(each_c_voxels)
                label.append(ind)

        if islabel is False:
            self.mean = np.mean(np.array(X)).astype(np.int32)

        return np.array(X), np.array(label)

    def save_featuremap(self, x, y, save_filepath):
        print("[*] Start Save Feature Map Mode")
        self.forward_only = True

        print("X to predict", x.shape, "on Save Feature Map Mode")
        print("label to predict", y.shape, "on Save Feature Map Mode")

        # mean subtraction
        x -= self.mean

        # print("DD", np.array(x).shape)
        # print("DD", np.array(y).shape)


        self.batch_size = 1
        x_shape = np.array(x).shape
        for ind in range(len(x)):

            batch_xs = np.reshape(x[ind], (1, x_shape[1], x_shape[2], x_shape[3], x_shape[4]))
            batch_ys = np.reshape(y[ind], (1))
            # print("DD", np.array(batch_xs).shape)
            # print("DD", np.array(batch_ys).shape)

            conv0, conv1, conv2 = self.session.run([self.feature_list[0],self.feature_list[1],self.feature_list[2]], feed_dict={self.X: batch_xs, self.Y: batch_ys, self.drop_rate: 1.0})
            os.makedirs(os.path.join(save_filepath, str(ind), "conv0"))
            os.makedirs(os.path.join(save_filepath, str(ind), "conv1"))
            os.makedirs(os.path.join(save_filepath, str(ind), "conv2"))
            conv0_s = np.array(conv0).shape
            conv1_s = np.array(conv1).shape
            conv2_s = np.array(conv2).shape

            conv0 = np.reshape(conv0, (conv0_s[1],conv0_s[2],conv0_s[3],conv0_s[4]))
            conv1 = np.reshape(conv1, (conv1_s[1], conv1_s[2], conv1_s[3], conv1_s[4]))
            conv2 = np.reshape(conv2, (conv2_s[1], conv2_s[2], conv2_s[3], conv2_s[4]))

            conv0 = np.transpose(conv0, axes=[3,0,1,2])
            conv1 = np.transpose(conv1, axes=[3, 0, 1, 2])
            conv2 = np.transpose(conv2, axes=[3, 0, 1, 2])

            for cind, vs in enumerate(conv0):
                viewer(vs, rows=8, cols=8, show_every=1, save_path=os.path.join(save_filepath, str(ind), "conv0", str(ind)+"_"+str(cind)+".png"))
            for cind, vs in enumerate(conv1):
                viewer(vs, rows=8, cols=8, show_every=1, save_path=os.path.join(save_filepath, str(ind), "conv1", str(ind)+"_"+str(cind)+".png"))
            for cind, vs in enumerate(conv2):
                viewer(vs, rows=8, cols=8, show_every=1, save_path=os.path.join(save_filepath, str(ind), "conv2", str(ind)+"_"+str(cind)+".png"))






























