import pydicom #read the dicom files
import os # do directory operations
import argparse
import pandas as pd #nice for data analysis
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import glob


def slice_it(li, num_of_chunks=2):
    start=0
    for i in range(num_of_chunks):
        stop = start + len(li[i::num_of_chunks]) # len(li[i::cols]) is the number remained from i to end on each cols step
        yield li[start:stop]
        start = stop

def mean(l):
    return sum(l)/len(l)

def resize_3D(voxel, dsize):
    """
    :param voxel: D, H, W
    :param dsize: size of voxel to change
    :return: voxels of which size is changed
    """

    origin_depth = voxel.shape[0]
    origin_height = voxel.shape[1]
    origin_width = voxel.shape[2]

    new_depth = dsize[0]
    new_height = dsize[1]
    new_width = dsize[2]


    new_slices = []
    for d_ in voxel:
        if new_height * new_width > origin_height * origin_width:
            new_slice = cv2.resize(d_, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            new_slice = cv2.resize(d_, (new_width, new_height), interpolation=cv2.INTER_AREA)
        new_slices.append(new_slice)

    instance = np.array(new_slices)
    #print("H, W changed", instance.shape)
    t_instance = np.transpose(instance, [1, 0, 2])

    new_slices = []
    for h_ in t_instance:

        if new_depth > origin_depth:
            new_slice = cv2.resize(h_, (new_width, new_depth),
                                   interpolation=cv2.INTER_CUBIC)  # (x, y) means y is row, x is column
        else:
            new_slice = cv2.resize(h_, (new_width, new_depth), interpolation=cv2.INTER_AREA)

        # plt.imshow(new_slice)
        # plt.show()
        new_slices.append(new_slice)
    t_instance = np.array(new_slices)
    #print("D changed", t_instance.shape)
    result = np.transpose(t_instance, [1, 0, 2])
    #print("result", result.shape)
    return result

#
def process_data(each_patient, IMG_PX_SIZE, HM_SLICES):
    '''
    :param patient: path of each patient_dir
    :param visualize: whether figures are needed
    :return:
    '''

    #slices = [pydicom.read_file(each_patient+'/'+s) for s in os.listdir(each_patient)]
    slices = [pydicom.read_file(s) for s in glob.glob(os.path.join(each_patient, "*.dcm"))]
    try:
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    except AttributeError as ae:
        import matplotlib.pyplot as plt
        print(each_patient, len(slices))
        pass

    slices = [np.array(each_slice.pixel_array) for each_slice in slices]
    slices = resize_3D(np.array(slices), (HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE))
    # new_slices = []
    # slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]
    #
    # for slice_chunk in slice_it(slices, HM_SLICES):
    #     new_slices.append(np.mean(slice_chunk,axis=0))

    return np.array(slices)  # I changed for the objects to use only pythonic list

def chkdir(path):
    '''
    check the path given as a ordinary directory
    :return: bool value if this path have garbage meaning positive
    '''
    if '.zip' in path:
        return True
    elif 'xlsx' in path or 'xls' in path or 'py' in path:
        return True
    return False

def dicom_loader(save_filename, data_dir=None, img_h_w=None, img_d=None, milestone=None):
    not_gray_file = []
    much_data = []  # list of class data
    p_label_data = []
    if os.path.isfile(save_filename) is True:
        print("that filename is already stored!")
    else:
        print("[*] Creating new npy file")
        # we will save big list array into a file.

        class_list = os.listdir(data_dir) # ['gr1_pet_list_fbb', 'gr2_pet_list_fbb', 'gr3_pet_list_fbb']

        for num, each_class in enumerate(class_list):
            print("DD", each_class) # nc
            p_label = []
            each_class_data = []
            if chkdir(each_class):
                continue
            for num, each_patient in enumerate(os.listdir(os.path.join(data_dir, each_class))):
                if num % 100 == 0:
                    print(num)
                try:

                    if milestone is not None:
                        try:

                            img_data = process_data(os.path.join(data_dir, each_class, each_patient, milestone), img_h_w, img_d)
                        except FileNotFoundError as e:
                            print(e)
                            continue
                    else :

                        img_data = process_data(os.path.join(data_dir, each_class, each_patient), img_h_w, img_d)
                        img_data = np.array(img_data)

                    p_instance = str(each_class)+"_"+each_patient # nc_xxxxxx
                    p_label.append(p_instance)

                    if len(img_data) < 20: # check integrity on neurology project
                        continue

                    if img_data.shape[-1] == 3:
                        not_gray_file.append(os.path.join(data_dir, each_class, each_patient))
                        print("chk", each_patient)
                        print("it's color space is 3", np.array(each_patient).shape)
                        pass
                    if img_data.shape[-1] != 1:
                        img_data = np.expand_dims(img_data, axis=len(img_data.shape))

                    if len(np.array(img_data).shape) < 4:
                        print("chk", each_patient)
                        print("it's shape", np.array(each_patient).shape)
                        pass
                    else :
                        each_class_data.append(np.array(img_data))
                    print(np.array(img_data).shape)

                except KeyError as e:
                    print('This is unlabeled data')
                    pass
                except NotADirectoryError as nade:
                    print(nade)
                    continue
                except AttributeError as atbe:
                    print(atbe)
                    print(os.path.join(data_dir, each_class, each_patient))
                    break
            much_data.append(np.array(each_class_data))
            p_label_data.append(np.array(p_label))

        print("muchdata",len(much_data))
        print("#########3",np.array(much_data).shape)
        print("#########3", np.array(much_data[0]).shape)
        print("#########3", np.array(much_data[1]).shape)
        #print("#########3", np.array(much_data[2]).shape)
        #print("#########3", np.array(much_data[0][0]).shape)
        np.save(save_filename, np.array(much_data))
        np.save(save_filename+"_p_label.npy", np.array(p_label_data))
    much_data = np.load(save_filename, encoding='latin1')
    p_label_data = np.load(save_filename+"_p_label.npy", encoding='latin1')
    return np.array(much_data), not_gray_file, np.array(p_label_data)


"""
convert dicom files into npy files to use them as voxels having only numbers.
if any set of dicom files were already converted into npy files, then doesn't work but print their shape for testing.

"""
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="dicom directory you need to process")
    parser.add_argument('--save_filename', type=str,
                        help="filename of numpy array that have dicoms pixels")
    parser.add_argument('--img_px_size', type=int, help="img size to resize")
    parser.add_argument('--slices', type=int, help="number of slices to norm")
    parser.add_argument('--milestone', default=None, type=str, help="")

    args = parser.parse_args()
    dl, _, p_label = dicom_loader(args.save_filename, args.data_dir, args.img_px_size, args.slices, args.milestone) # [dataset, not_gray_file_list]

    #print("saved file shape : ", dl.shape)
    print("saved file shape : ", dl[0].shape)
    #print("#########3", np.array(much_data).shape)

    print("#########3 p_label", np.array(p_label).shape)
    print("#########3 p_label", p_label[0])
    #print("#########3", dl[1])
    print("#########3", np.array(dl[0]).shape)
    #print("#########3", np.array(dl[0]).shape)
    # print("#########3", np.array(dl[0]).shape)

    print("#########3", np.array(dl[0]).shape)


    #print("#########3", np.array(dl[0][2]).shape)
    #print("gray file list : ", dl[1])

if __name__=="__main__":
    main()


