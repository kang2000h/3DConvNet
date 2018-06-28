import tensorflow as tf

import os
from Conv3DNet import *
from utils import *
from dicom_loader import dicom_loader
#from dicom_loader2 import dicom_loader
from Conv3DNet import Conv3DNet
from sklearn.model_selection import LeaveOneOut


flags = tf.app.flags
flags.DEFINE_string("mode", None, "define what to do")
flags.DEFINE_string("data_path", "./data/muchdata-50-50-20.npy", "data_path to use")
flags.DEFINE_integer("batch_size", 20, "batch_size of model")
flags.DEFINE_integer("f_filter", 32, "number of first filter")
flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
flags.DEFINE_string("tv_type", "holdout", "whether to do holdout or CV")
flags.DEFINE_float("train_rate", 0.7, "how many take the samples as a training data")
flags.DEFINE_float("loss_threshold", None, "threshold of loss your model is supposed to be trained")
flags.DEFINE_float("learning_rate", 0.0001, "how many take the samples as a training data")
flags.DEFINE_integer("epoch", 50, "epoch to train")
flags.DEFINE_string("model_dir", None, "model_dir to load")
flags.DEFINE_string("checkpoint_name", "model.ckpt","name of checkpoint to save")
flags.DEFINE_float("learning_rate_decay_factor",0.99 ,"learning_rate_decay_factor")
flags.DEFINE_string("lr_decay","normal" ,"type of learning rate decay")

flags.DEFINE_boolean("is_transfer",False ,"whether to train as a transfer learning")

# flags.DEFINE_integer("depth", 128, "size of depth of voxels to resize")
# flags.DEFINE_integer("height", 128, "size of height of voxels to resize")
# flags.DEFINE_integer("width", 128, "size of width of voxels to resize")

flags.DEFINE_boolean("visualize",False ,"whether to show the variance of Accuracy on Train/Val phase")

FLAGS = flags.FLAGS
def main(_):
    pp.pprint(flags.FLAGS.__flags)

    ## change for Error Analysis
    much_data = dicom_loader(FLAGS.data_path)  # num_classes, num_data, depth, height, width -> not good, we need to be robust on even inbalanced classes
    # much_data exported from dicom_lodear2 shaped like [3,]
    #  much_data is [2, ] # [much_data, not_gray_file_list]

    #p_label = np.array(much_data[2])
    much_data = np.array(much_data[0])  # ideally, NC, N, D, H, W, C
    ##

    much_data, _ = dicom_loader(FLAGS.data_path)  # num_classes, num_data, depth, height, width -> not good, we need to be robust on even inbalanced classes
    # much_data is [2, ] # [much_data, not_gray_file_list]

    much_data = np.array(much_data) # ideally, NC, N, D, H, W, C

    print("Input_data", much_data.shape)
    print("Input_data", np.array(much_data[0]).shape)
    print("Input_data", np.array(much_data[1]).shape)

    num_classes = len(much_data)

    num_total_data = np.array([len(dn) for dn in much_data]).sum()
    class_weights = [(1-(len(dn)/num_total_data)) for dn in much_data]
    #class_weights = None
    print("num_total_data", num_total_data, class_weights)

    c_sample = np.array(much_data[0]) # pick any class to specify D, H, W info

    print(np.array(c_sample[0]).shape)

    if len(c_sample.shape)<5: # it need to be 5 (N, D, H, W, C)
        print("channel dimension doesn't seem to be specified")
        temp = []
        for ind, c in enumerate(much_data):
            temp.append(np.expand_dims(much_data[ind], axis=len(np.array(c).shape)))
        much_data = np.array(temp)
        c_sample = np.expand_dims(c_sample, axis=len(c_sample.shape))

    #print("??",c_sample.shape)
    #print("Data loaded completely!", np.array(c_sample).shape)

    input_depth = c_sample[0].shape[0]
    input_height = c_sample[0].shape[1]
    input_width = c_sample[0].shape[2]
    input_cdim = c_sample[0].shape[3]

    #os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
    run_config.gpu_options.allow_growth = True #allocate dynamically

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # run_config = tf.ConfigProto(gpu_options=gpu_options)
    #with tf.Session() as sess:
    with tf.Session(config=run_config) as sess:
        if FLAGS.mode == 'train':

            forward_only = False
            model = Conv3DNet(sess, input_depth, input_height, input_width, input_cdim, num_classes, FLAGS.batch_size, FLAGS.epoch, FLAGS.model_dir,FLAGS.checkpoint_name,
                              lr_decay=FLAGS.lr_decay, visualize=FLAGS.visualize, learning_rate=FLAGS.learning_rate, learning_rate_decay_factor=FLAGS.learning_rate_decay_factor, train_rate=FLAGS.train_rate, tv_type = FLAGS.tv_type,
                              class_weights=class_weights, f_filter = FLAGS.f_filter, beta1 = FLAGS.beta1, forward_only=forward_only, transfer_learning=FLAGS.is_transfer)

            model.create_model() # Structuring proper tensor graph

            print("[*] Enter Train Phase...")
            x, y = model.get_batch_v2(much_data)

            _ = model.train_v2(x, y, opserve_v=False, save_mode=True, phase_name="Train_Phase", loss_threshold=FLAGS.loss_threshold)
            #model.train(x, y)
            #model.train_v2(np.array(x), np.array(y), split=True, save_mode=True)

            alarm('"Training has finished!"', voice=True)

        elif FLAGS.mode=='val':
            if FLAGS.tv_type=='holdout' or FLAGS.tv_type=="ho":

                forward_only = False
                model = Conv3DNet(sess, input_depth, input_height, input_width, input_cdim, num_classes,
                                  FLAGS.batch_size, FLAGS.epoch, FLAGS.model_dir, FLAGS.checkpoint_name,
                                  lr_decay=FLAGS.lr_decay, visualize=FLAGS.visualize, learning_rate=FLAGS.learning_rate,
                                  learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                                  train_rate=FLAGS.train_rate, tv_type=FLAGS.tv_type,
                                  class_weights=class_weights, f_filter=FLAGS.f_filter, beta1=FLAGS.beta1,
                                  forward_only=forward_only, transfer_learning=FLAGS.is_transfer)

                #model.create_model()  # Structuring proper tensor graph
                model.create_model()  # Structuring proper tensor graph
                #x, y = model.get_batch(much_data, mode='val', depth=FLAGS.depth, height=FzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzLAGS.height, width=FLAGS.width)
                x, y = model.get_batch_v2(much_data)
                train_rate = 0.7
                num_division = np.array(train_rate * len(x)).astype(np.int32)
                train_data, val_data = subsampling([x, y], num_division, option='multi_source')
                _ = model.train_v2(train_data[0], train_data[1], val_data[0], val_data[1], opserve_v=True, save_mode=False, loss_threshold=FLAGS.loss_threshold) # 0 index means 0.7 of each data, i.e. train data
                val_acc, y_label, pred, _ =  model.predict(val_data[0], val_data[1], "Val Phase", roc_ft_extract=True)

                alarm("Validation has finished!", voice=True)

            elif FLAGS.tv_type=='cv' or FLAGS.tv_type=="CV":
                forward_only = False
                model = Conv3DNet(sess, input_depth, input_height, input_width, input_cdim, num_classes,
                                  FLAGS.batch_size, FLAGS.epoch, FLAGS.model_dir, FLAGS.checkpoint_name,
                                  lr_decay=FLAGS.lr_decay, visualize=FLAGS.visualize, learning_rate=FLAGS.learning_rate,
                                  learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                                  train_rate=FLAGS.train_rate, tv_type=FLAGS.tv_type,
                                  class_weights=class_weights, f_filter=FLAGS.f_filter, beta1=FLAGS.beta1,
                                  forward_only=forward_only, transfer_learning=FLAGS.is_transfer)

                model.create_model()  # Structuring proper tensor graph
                x, y = model.get_batch_v2(much_data)

                # K fold Cross validation
                num_K = 10
                cv = StratifiedKFold(n_splits=num_K, shuffle=True)
                cv.get_n_splits(x, y)
                sess.run(tf.global_variables_initializer())

                val_acc_list = []
                for ind, (train_index, test_index) in enumerate(cv.split(x, y)):
                    print("[*] ", ind, " CV loop")
                    sess.run(tf.global_variables_initializer())

                    _ = model.train_v2(x[train_index], y[train_index], x[test_index], y[test_index], opserve_v=True, save_mode=False, phase_name=str(ind), loss_threshold=FLAGS.loss_threshold)
                    val_acc, y_label, pred, _ = model.predict(x[test_index], y[test_index], phase_name="_"+str(ind)+"_CV loop", roc_ft_extract=True)
                    val_acc_list.append(val_acc)
                plot_errorbar(val_acc_list, "./results/CV_ACC_ERRORBAR.png")
                print("Total CV Accuracy Mean : ", np.array(val_acc_list).mean())
                with open("./results/CV_val_acc_result_list.txt", "w") as f:
                    for item in val_acc_list:
                        f.write("%s\n"%item)
                    f.write("Mean %s\n" % np.array(val_acc_list).mean())


        elif FLAGS.mode== "test" or FLAGS.mode=='predict':
            forward_only = True
            model = Conv3DNet(sess, input_depth, input_height, input_width, input_cdim, num_classes,
                              FLAGS.batch_size, FLAGS.epoch, FLAGS.model_dir, FLAGS.checkpoint_name,
                              lr_decay=FLAGS.lr_decay, visualize=FLAGS.visualize, learning_rate=FLAGS.learning_rate,
                              learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                              train_rate=FLAGS.train_rate, tv_type=FLAGS.tv_type,
                              class_weights=class_weights, f_filter=FLAGS.f_filter, beta1=FLAGS.beta1,
                              forward_only=forward_only, transfer_learning=False)

            model.create_model()  # Structuring proper tensor graph

            print("[*] Entering Predict Phase...")
            x, y = model.get_batch_v2(much_data)

            val_acc, y_label, pred, v_feature = model.predict(x, y, phase_name="Test Phase", roc_ft_extract=True)
            concatlist = concatList_axis_1([pred, y_label])
            save_list_info("./results/ROC_data_test.csv", concatlist)
            concatlist = concatList_axis_1([v_feature, y_label])
            save_list_info("./results/test_feature.csv", concatlist)
            #alarm("Test has finished!", voice=True)

            with open("./results/holdout_test_acc_result.txt", "w") as f:
                f.write("Acc %s\n" % np.array(val_acc))

            print("[!] Test is finished")

        elif FLAGS.mode == "loocv_eval" or FLAGS.mode == "le":
            forward_only = False
            model = Conv3DNet(sess, input_depth, input_height, input_width, input_cdim, num_classes,
                              FLAGS.batch_size, FLAGS.epoch, FLAGS.model_dir, FLAGS.checkpoint_name,
                              lr_decay=FLAGS.lr_decay, visualize=FLAGS.visualize, learning_rate=FLAGS.learning_rate,
                              learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                              train_rate=FLAGS.train_rate, tv_type=FLAGS.tv_type,
                              class_weights=class_weights, f_filter=FLAGS.f_filter, beta1=FLAGS.beta1,
                              forward_only=forward_only, transfer_learning=False)

            model.create_model()  # Structuring proper tensor graph
            x, y = model.get_batch_v2(much_data)

            # K fold Cross validation
            num_K = len(x)
            #cv = StratifiedKFold(n_splits=num_K, shuffle=True)
            # cv.get_n_splits(x, y)
            loo = LeaveOneOut()
            loo.get_n_splits(x)

            sess.run(tf.global_variables_initializer())

            val_acc_list = []
            #for ind, (train_index, test_index) in enumerate(cv.split(x, y)):
            for ind, (train_index, test_index) in enumerate(loo.split(x)):
                print("[*] ", ind+1,"/", num_K," CV loop")
                sess.run(tf.global_variables_initializer())

                _ = model.train_v2(x[train_index], y[train_index], x[test_index], y[test_index], opserve_v=False, save_mode=False, phase_name=str(ind), loss_threshold=FLAGS.loss_threshold)
                val_acc, y_label, pred, _ = model.predict(x[test_index], y[test_index],
                                                          phase_name="_" + str(ind) + "_CV loop", roc_ft_extract=False, is_loocv=True)
                val_acc_list.append(val_acc)
            plot_errorbar(val_acc_list, "./results/CV_ACC_ERRORBAR.png")
            print("Total LOOCV Accuracy Mean : ", np.array(val_acc_list).mean())
            with open("./results/LOOCV_val_acc_result_list.txt", "w") as f:
                for item in val_acc_list:
                    f.write("%s\n" % item)
                f.write("Mean %s\n" % np.array(val_acc_list).mean())
        elif FLAGS.mode == "feature_analysis" or FLAGS.mode == "fa":
            forward_only = True
            model = Conv3DNet(sess, input_depth, input_height, input_width, input_cdim, num_classes,
                              FLAGS.batch_size, FLAGS.epoch, FLAGS.model_dir, FLAGS.checkpoint_name,
                              lr_decay=FLAGS.lr_decay, visualize=FLAGS.visualize, learning_rate=FLAGS.learning_rate,
                              learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                              train_rate=FLAGS.train_rate, tv_type=FLAGS.tv_type,
                              class_weights=class_weights, f_filter=FLAGS.f_filter, beta1=FLAGS.beta1,
                              forward_only=forward_only, transfer_learning=False)

            model.create_model()  # Structuring proper tensor graph
            x, y = model.get_batch_v2(much_data)

            sess.run(tf.global_variables_initializer())

            # train_rate = 0.7
            # num_division = np.array(train_rate * len(x)).astype(np.int32)
            # train_data, val_data = subsampling([x, y], num_division, option='multi_source')
            # _ = model.train_v2(train_data[0], train_data[1], val_data[0], val_data[1], opserve_v=False,
            #                    save_mode=False)  # 0 index means 0.7 of each data, i.e. train data
            model.save_featuremap(x, y, "./results/feature_analysis")







if __name__=="__main__":
    tf.app.run()