import os # do directory operations
import argparse
import pandas as pd #nice for data analysis
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

import dicom2nifti
import pydicom #read the dicom files
import nibabel as nib


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

def get_full_dir(upper_dir):
    lower_dir = os.listdir(upper_dir)
    lower_full_dir = [os.path.join(upper_dir, c) for c in lower_dir]
    return lower_full_dir


"""
convert dicom files into npy files to use them as voxels having only numbers.
if any set of dicom files were already converted into npy files, then doesn't work but print their shape for testing.

"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="dicom directory you need to process, This dir is parent dir of each patient dir")
    #parser.add_argument('--save_filename', type=str, help="filename of numpy array that you will save as a output of this program")
    parser.add_argument('--output_dir', type=str, help="directory name that you will save as a output of this program")

    args = parser.parse_args()
    full_p_dir = get_full_dir(args.data_dir)

    for p in full_p_dir:
        if chkdir(p):
            continue
        else:
            p_name = p.split("/")[-1]
            dicom2nifti.dicom_series_to_nifti(p, os.path.join(args.output_dir, p_name+".nii"), reorient_nifti=True)


if __name__=="__main__":
    main()


