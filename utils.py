import pprint
import numpy as np
import tensorflow as tf
import csv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter()

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def conv3d(x, output_channel, k_d=5, k_h=5, k_w=5, s_d=2, s_h=2, s_w=2, stddev=0.02, padding = 'VALID', name="conv3d"):
    with tf.variable_scope(name) as scope:
        #w = tf.get_variable('w', [k_d, k_h, k_w, x.get_shape()[-1], output_channel],
        #                    initializer=tf.truncated_normal_initializer(stddev=stddev))
        w = tf.get_variable('w', [k_d, k_h, k_w, x.get_shape()[-1], output_channel],
                                               initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv3d(x, w, strides=[1, s_d, s_h, s_w, 1], padding=padding)
        #biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))
        biases = tf.get_variable('biases', [output_channel], initializer=tf.contrib.layers.xavier_initializer())
        #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

    return conv

def lrelu(x, leak=0.1, name="lrelu"):
    return tf.maximum(x, leak*x)

def dropout(x, drop_rate=0.5, istrain=False):
    return tf.layers.dropout(x, rate=drop_rate, training=istrain)

def linear(input_, output_size, scope=None, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32,
                             tf.contrib.layers.xavier_initializer(False))
        bias = tf.get_variable("bias", [output_size], initializer=tf.contrib.layers.xavier_initializer(False))
        if with_w:
            return lrelu(tf.matmul(input_, matrix) + bias, matrix, bias, leak=0.1)
        else :
            return lrelu(tf.matmul(input_, matrix) + bias, leak=0.1)


# we need to define the stride of depth and size of filter compared to 2d Conv
def maxpool3d(x, k_d=2, k_h=2, k_w=2, s_d=2, s_h=2, s_w=2, padding='VALID',name="maxpool3d"):
    #                        size of window         movement of window
    return tf.nn.max_pool3d(x, ksize=[1, k_d, k_h, k_w, 1], strides=[1, s_d, s_h, s_w, 1], padding=padding)



class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def alarm(content=None, voice=False):
    import os
    duration = 1  # second
    freq = 440  # Hz
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    if not voice:
        # sudo apt install sox # On Debian/Ubuntu/LinuxMint
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    elif voice:
        # sudo apt install speech-dispatcher
        os.system('spd-say '+content)

def da_processor_3d(sess, datas): # (B, D, H, W, C)

    # from collections import defaultdict
    # from gc import get_objects
    # before = defaultdict(int)
    # after = defaultdict(int)
    # for i in get_objects():
    #     before[type(i)] += 1


    # that's inefficient to declare tensor on each iteration.
    # whenever declaring tensor, many of list or dict are generated

    rotate_input = tf.placeholder(tf.float32, [datas.shape[1], datas.shape[2], datas.shape[3], datas.shape[4]])
    flip_input = tf.placeholder(tf.float32, [datas.shape[2], datas.shape[3], datas.shape[4]])

    angle = tf.placeholder(tf.float32, shape=())
    rotated_img = tf.contrib.image.rotate(rotate_input, angles=angle, interpolation='BILINEAR')

    flipped_ud_img = tf.image.flip_up_down(flip_input)
    flipped_rl_img = tf.image.flip_left_right(flip_input)

    result = []
    for ind in range(datas.shape[0]):
        # c = np.random.randint(3)
        r = np.random.rand() # random real value between 0 and 1

        #if r < 0.6: # normal
        if r < 0.0:
            result.append(datas[ind]) # add (D, H, W, C)

        #elif r < 0.8: # rotate
        elif r < 0.5:  # rotate
            #ran_angle = np.random.uniform(0, np.pi*2)
            ran_angle = np.random.uniform(0, np.pi * 2/3)*3
            rotated_img_ = sess.run(rotated_img, feed_dict={rotate_input:datas[ind], angle:ran_angle})
            result.append(np.array(rotated_img_))

        else: # flip
            fc = np.random.randint(2)
            flipped = []
            batch = datas[ind]
            if fc is 0: # up_down
                for ind_d in range(datas.shape[1]):
                    flipped_img_ = sess.run(flipped_ud_img, feed_dict={flip_input:batch[ind]})
                    flipped.append(np.array(flipped_img_))
                result.append(np.stack(flipped, axis=0))


            else: # left_right
                for ind_d in range(datas.shape[1]):
                    flipped_img_ = sess.run(flipped_rl_img, feed_dict={flip_input:batch[ind]})
                    flipped.append(np.array(flipped_img_))
                result.append(np.stack(flipped, axis=0))
    result = np.stack(result, axis=0)

    # Check memory leak by showing running data structure
    # import objgraph
    # import inspect, random
    # print("####################################")
    # print(objgraph.show_growth(limit=10))


    # Check how much data structure is running
    # print("#################################")
    # for i in get_objects():
    #     after[type(i)] += 1
    # print([(k, after[k] - before[k]) for k in after if after[k] - before[k]])
    # after = defaultdict(int)

    return result


# dicom_loader에 보강할 필요가 있어보임
'''
import numpy as np
much_data = np.load('./data/muchdata-50-50-20.npy') # 1397, 2 -> num_classes, num_data, depth, height, width, channel=1

# print(much_data.shape)
# print(np.array(much_data[0][0]).shape)
# print(much_data[0][1])

data = np.array([cont[0] for cont in much_data])
label = np.array([cont[1] for cont in much_data])

#data = np.expand_dims(data, axis=len(data.shape))
print(data.shape)


label = np.array([np.argmax(cont) for cont in label])
print(label.shape)

tmp = [[] for _ in range(2)]
for ind, val in enumerate(label):
    tmp[val].append(np.array(data[ind]))

tmp = np.array(tmp)
print(tmp.shape)
print(np.array(tmp[0]).shape)
print(np.array(tmp[1]).shape)
#np.save('./data/kaggle_data.npy',tmp)
'''

# converting our fbb dicom data into voxels
# import os
# import glob
# import numpy as np
#
# from dicom_loader import *
# from dicom_viewer import *
#
# data_dir ='/media/donga/Deep/entire_pet'
# #classes = os.listdir(filename)
# #patient = os.listdir(os.path.join(data_dir, classes[0]))
# milestone = 'fbbstatic'
# '''
# much_data = [[] for _ in range(len(classes))]
# for c in classes:
#     patients = os.listdir(os.path.join(filename, c))
#     for patient in patients:
#         p_dcms = load_scan(os.path.join(filename, c, patient, milestone))
#         p_slices = get_pixels_hu(p_dcms)
#         viewer(p_slices)
# '''
# save_filename = './data/amyloid_pet_20_50_50.npy'
#
# much_data, _ = dicom_loader(save_filename, data_dir, img_h_w=50, img_d=20, milestone=milestone)
# much_data = np.array(much_data)
#
# for ind, d in enumerate(much_data):
#     print(ind+1, len(much_data[ind]))
#     for instance in d:
#         print(instance.shape)



def draw_roc(y_score, label, class_to_draw, title='Receiver operating characteristic example', color='darkorange'):
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp


    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = label.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], threshold = roc_curve(label[:, i], y_score[:,i])  # fpr, tpr of ith class, when y_test[:,i] is only 1, the internal list is appended
        roc_auc[i] = auc(fpr[i], tpr[i])

    # print(y_test) # test set of dataset
    # print(np.array(tpr).shape)

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[class_to_draw], tpr[class_to_draw], color, lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[class_to_draw])

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    plt.gcf().clear()
    plt.clf()
    plt.cla()
    plt.close()

# plot_roc_curve(fpr[ind], tpr[ind], label='ROC curve (area = %0.2f)' % roc_auc[ind])
def plot_roc_curve(y_label, y_score, label=None, filepath=None):

    num_class = np.array(y_label).shape[1]

    for nc in range(num_class):

        fpr, tpr, _ = roc_curve(y_label[nc], y_score[nc])
        roc_auc = auc(fpr, tpr)
        if label is None:
            label = "area=%0.2f" % roc_auc
        else:
            label = label + " (area=%0.2f)"%roc_auc
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0,1,0,1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(filepath+"_"+str(nc)+".png")
        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()
        #plt.show()

def save_list_info(savefilename, linfo, encoding='utf-8', newline=''):

    '''
    :param     savefilename : name of file to save
    :param     linfo : list of information to save_list_info, on row-major order,
                e.g [[1,"kim JS", False],[2, "Park SM", True],...]

    :return: none
    '''

    with open(savefilename, 'w', encoding=encoding, newline=newline) as f:
        wr = csv.writer(f)

        for r in linfo:
            wr.writerow(r)

def concatList_axis_1(llist):
    '''
    :param llist: list of list, each current row need to be column as a property, e.g. [[1,0,0],[0,1,0], ... ]
           to save csv file and analyse datas using the file
    :return: list of list, row-majored lists
    '''
    return np.concatenate(llist, axis=1)


def plot_tv_trend(filepath, train_accs, val_accs):
    import matplotlib.pyplot as plt
    import matplotlib
    # matplotlib.rcParams.update({'font.size':22})    #
    # matplotlib.rc('font', size=23)
    # matplotlib.rc('axes', labelsize=25)
    # matplotlib.rc('legend', fontsize=25)

    x = np.arange(len(train_accs))
    y1 = np.array(train_accs) * 100
    y2 = np.array(val_accs) * 100

    plt.plot(x, y1, label="Train")
    plt.plot(x, y2, linestyle="--", label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.savefig(filepath)
    #plt.show()
    plt.gcf().clear()
    plt.clf()
    plt.cla()
    plt.close()


def _extract(cond, ndarray):
    if len(cond)!=len(ndarray):
        raise chk_integrity("chk length of cond and ndarray, they need to have same length.")

    extracted = []
    for ind, factor in enumerate(ndarray):
        if bool(cond[ind]) is True:

            extracted.append(factor)
    return np.array(extracted)

def subsampling(source, num_T, option=None):
    """

    :param source: a list or list of multi array that you're suppose to divide into 2 parts with same indexing
    :param num_T: size of a part of data(fore_data)
    :param option: if you're suppose to divide list of multi array like features and label then you can input 'multi_source' into this argument
    :return: a part of data diveded and another data(fore_data and back_data)
    """

    if option=='multi_source':
        fore_data = []
        back_data = []
        for ss in source :
            ss = np.array(ss)

            f_mask = np.zeros(len(ss))
            f_mask_ind = np.random.choice(len(f_mask), num_T, replace=False)
            for fm_ind in f_mask_ind:
                f_mask[fm_ind] = 1

            # f_masked fore dataset
            f_cond = np.equal(f_mask, 1) == True
            f_data = _extract(f_cond, ss)
            fore_data.append(f_data)

            # v_masked validation set
            reversed = np.where(f_mask > 0.5, 0, 1)
            b_cond = np.equal(reversed, 1) == True
            b_data = _extract(b_cond, ss)
            back_data.append(b_data)
        return fore_data, back_data

    else :
        source = np.array(source)
        t_mask = np.zeros(len(source))
        t_mask_ind = np.random.choice(len(t_mask), num_T, replace=False)
        for tm_ind in t_mask_ind:
            t_mask[tm_ind]=1

        # t_masked train set
        t_cond = np.equal(t_mask, 1)==True
        train_data = np.extract(t_cond, source)

        # v_masked validation set
        reversed = np.where(t_mask>0.5, 0, 1)
        v_cond = np.equal(reversed, 1)==True
        val_data = np.extract(v_cond, source)
        return train_data, val_data


def plot_errorbar(acc_list, filepath):
    param = [0]
    print("acc_list", acc_list)
    acc_list = np.array(acc_list)
    acc_mean = acc_list.mean()
    acc_std = acc_list.std()
    acc_stderr = acc_std / np.sqrt(len(acc_list))
    with plt.style.context(('seaborn-poster')):
        plt.clf()
        plt.cla()
        ax = plt.subplot(111)
        # ax.set_xscale('log')
        ax.errorbar(param, acc_mean, yerr=acc_stderr, marker='o', alpha=0.8, ecolor='black',
                    elinewidth=2)
        plt.plot(param * len(acc_list), acc_list, "ro")
        plt.ylim([0.0, 1.0])
        # plt.xlim([0.0001, 100000.0])

        my_xticks = ['CV']
        plt.xticks(param, my_xticks)

        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.show()

        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()

def viewer(samples, rows=5, cols=5, show_every=5, save_path=None):

    print("instance shape",samples.shape)
    fig = plt.figure()
    for ind, each_slices in enumerate(samples):

        #print("each_slices shape", each_slices.shape)
        ind+=1
        if ind/show_every > rows*cols:
            break
        if ind%show_every==0:
            y = fig.add_subplot(rows, cols, ind/show_every)  # it will be [3, 4] sub plot grid
            origin_shape = each_slices.shape
            y.imshow(np.reshape(each_slices, (origin_shape[0], origin_shape[1])), cmap='gray')
    if save_path is not None:
        plt.savefig(save_path)
    else :
        plt.show()


def feature_scaling(train_feature, test_feature):
    # 훈련 세트에서 특성별 최솟값 계산
    min_on_training = train_feature.min(axis=0)
    # 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
    range_on_training = (train_feature - min_on_training).max(axis=0)

    # 훈련 데이터에 최솟값을 빼고 범위로 나누면
    # 각 특성에 대해 최솟값은 0, 최대값은 1입니다.
    train_x_scaled = (train_feature - min_on_training) / range_on_training
    #print("특성별 최소 값\n{}".format(train_x_scaled.min(axis=0)))
    #print("특성별 최대 값\n {}".format(train_x_scaled.max(axis=0)))

    # 테스트 세트에도 같은 작업을 적용하지만
    # 훈련 세트에서 계산한 최솟값과 범위를 사용합니다
    test_x_scaled = (test_feature - min_on_training) / range_on_training

    return train_x_scaled, test_x_scaled