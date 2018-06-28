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


    print("val_data generated", np.array(val_data).shape)
    return train_data, val_data

def main():
    np.random.seed(int(round(time.time())) % 1000)
    source_data = np.load(args.npy_filepath) # (num_classes, cN, D, H, W, 1)

    # for ind, sd in enumerate(source_data):
    #     print(ind, "label shape, ",np.array(sd).shape)

    division_ratio = np.array(args.ratio)


    num_classes = len(source_data)

    if args.mode=='tv' or args.mode=='TV':
        train = []
        val = []
        for c_data in source_data:
            num_division = np.array(division_ratio * len(c_data)).astype(np.int32)

            t, v = subsampling(c_data, num_division[0])
            train.append(t)
            val.append(v)

        # train = np.concatenate(train)
        # val = np.concatenate(val)

        print("divided train", np.array(train).shape)
        print("divided val", np.array(val).shape)
        np.save(os.path.join(args.output_directory, "_train.npy"), train)
        np.save(os.path.join(args.output_directory, "_val.npy"), val)
        return

    elif args.mode=='tvt' or args.mode=='TVT':
        train = []
        val = []
        test = []

        for c_data in source_data:
            num_division = division_ratio * len(c_data)

            tr, v = subsampling(c, num_division[0])
            train.append(t)
            v, te = subsampling(v, num_division[1])
            val.append(v)
            test.append(te)

        train = np.concatenate(train)
        val = np.concatenate(val)
        test = np.concatenate(test)

        print("divided train", train.shape)
        print("divided val", val.shape)
        print("divided test", test.shape)

        np.save(os.path.join(args.output_directory, "_train.npy"), train)
        np.save(os.path.join(args.output_directory, "_val.npy"), val)
        np.save(os.path.join(args.output_directory, "_test.npy"), test)

        return
    else :
        raise chk_integrity()

if __name__=="__main__":
    main()


