import os
import time
import argparse
import numpy as np

from exceptions import *

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, help="how to split")
parser.add_argument('--npy_filepath', type=str, help="filepath to export some information to draw roc curve")
parser.add_argument('--p_filepath', type=str, help="filepath to import patient ID to identify numpy voxels for error analysis")
parser.add_argument('--ratio', type=float, nargs='+', help="ratio of division")
parser.add_argument('--output_directory', type=str, help="directory name to save voxels for each cases(tv or tvt)")


args = parser.parse_args()

def _extract(cond, ndarray):
    if len(cond)!=len(ndarray):
        raise chk_integrity("chk length of cond and ndarray, they need to have same length.")

    extracted = []
    for ind, factor in enumerate(ndarray):
        if bool(cond[ind]) is True:
            extracted.append(factor)
    return np.array(extracted)

def subsampling(source, num_T):
    source = np.array(source) # (cN, D, W, H, 1)

    t_mask = np.zeros(len(source))
    t_mask_ind = np.random.choice(len(t_mask), num_T, replace=False)
    for tm_ind in t_mask_ind:
        t_mask[tm_ind]=1

    # t_masked train set
    t_cond = np.equal(t_mask, 1)==True
    train_data = _extract(t_cond, source)
    print("train_data generated", np.array(train_data).shape)

    # v_masked validation set
    reversed = np.where(t_mask>0.5, 0, 1)
    v_cond = np.equal(reversed, 1)==True
    val_data = _extract(v_cond, source)


    print("next_data generated", np.array(val_data).shape)
    return train_data, val_data

def main():
    np.random.seed(int(round(time.time())) % 1000)
    source_data = np.load(args.npy_filepath) # (num_classes, cN, D, H, W, 1)

    p_label_data = np.load("." + "".join(args.npy_filepath.split(".")[:-1])+ "_p_label.npy")  # (num_classes, cN)

    # for ind, sd in enumerate(source_data):
    #     print(ind, "label shape, ",np.array(sd).shape)

    division_ratio = np.array(args.ratio)

    num_classes = len(source_data)

    if args.mode=='tt' or args.mode=='TT':
        train = []
        test = []

        train_p_label=[]
        test_p_label=[]
        for c_data, c_p_label in zip(source_data, p_label_data):
            num_division = np.array(division_ratio * len(c_data)).astype(np.int32)

            tr, te = subsampling(c_data, num_division[0])
            tr_p_label, te_p_label = subsampling(c_p_label, num_division[0])

            train.append(tr)
            test.append(te)

            train_p_label.append(tr_p_label)
            test_p_label.append(te_p_label)

        # train = np.concatenate(train)
        # val = np.concatenate(val)

        print("divided train", np.array(train).shape)
        print("divided test", np.array(test).shape)
        print("divided train_p_label", np.array(train_p_label).shape)
        print("divided test_p_label", np.array(test_p_label).shape)

        np.save(os.path.join(args.output_directory, "_train.npy"), train)
        np.save(os.path.join(args.output_directory, "_test.npy"), test)
        np.save(os.path.join(args.output_directory, "_train_p_label.npy"), train_p_label)
        np.save(os.path.join(args.output_directory, "_test_p_label.npy"), test_p_label)
        return

    elif args.mode == 'tvt' or args.mode == 'TVT':
        train = []
        val = []
        test = []

        train_p_label = []
        val_p_label = []
        test_p_label = []

        for c_data, c_p_label in zip(source_data, p_label_data):
            num_division = division_ratio * len(c_data)

            tr, v = subsampling(c_data, num_division[0])
            train.append(tr)
            v, te = subsampling(v, num_division[1])
            val.append(v)
            test.append(te)

            tr_p_label, v_p_label = subsampling(c_p_label, num_division[0])
            train_p_label.append(tr_p_label)
            v_p_label, te_p_label = subsampling(v_p_label, num_division[1])
            val_p_label.append(v_p_label)
            test_p_label.append(te_p_label)

        # train = np.concatenate(train)
        # val = np.concatenate(val)
        # test = np.concatenate(test)

        train_p_label = np.concatenate(train_p_label)
        val_p_label = np.concatenate(val_p_label)
        test_p_label = np.concatenate(test_p_label)

        print("divided train", train.shape)
        print("divided val", val.shape)
        print("divided test", test.shape)

        print("divided train_p_label", train_p_label.shape)
        print("divided val_p_label", val_p_label.shape)
        print("divided test_p_label", test_p_label.shape)

        np.save(os.path.join(args.output_directory, "_train.npy"), train)
        np.save(os.path.join(args.output_directory, "_val.npy"), val)
        np.save(os.path.join(args.output_directory, "_test.npy"), test)

        np.save(os.path.join(args.output_directory, "_train_p_label.npy"), train_p_label)
        np.save(os.path.join(args.output_directory, "_val_p_label.npy"), val_p_label)
        np.save(os.path.join(args.output_directory, "_test_p_label.npy"), test_p_label)

        return
    else:
        raise chk_integrity()


if __name__ == "__main__":
    main()


