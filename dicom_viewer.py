### Reading DICOM Files...
import os
from glob import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt

import argparse



### Helper Functions ###
def load_scan(path):
    # Loop over the image files and store everything into a listd.
    """
    :param path: list of paths and its subdir will be refered
    :return: SliceThickness를 결정한 slices
    """

    slices = [dicom.read_file(path + '/' +s) for s in os.listdir(path) if "dcm" in s] #listdir은 dir을 받으면 안의 파일들의 리스트를 반환.
    slices.sort(key = lambda x : int(x.InstanceNumber))

    # try:
    #     slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2])
    # except: # what except? 아마 값이 없는 경우도 있나봄.
    #     slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    #
    # # each dicom file object seem to have some attr such as ImagePositionPatient, SliceLocation, SliceThickness
    # for s in slices:
    #     s.SliceThickness = slice_thickness #SliceThickness가 dicom객체의 attr로 존재하는 듯하다.
    return slices


def get_pixels_hu(scans):
    import numpy as np
    image = np.stack([s.pixel_array for s in scans]) # pixel_array는 dicom파일의 픽셀들을 볼 수 있다.

    # Convert to int16 (from sometimes int16),
    # Should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to HounsField units (HU)
    # intercept = scans[0].RescaleIntercept
    # slope = scans[0].RescaleSlope
    #
    # if slope != 1:
    #     image = slope* image.astype(np.float64)
    #     image = image.astype(np.int16)
    # image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


### Displaying Images
#Lets now create a histogram of all the voxel data in the study.
# import numpy as np
# import matplotlib.pyplot as plt
# id=1
# file_used = output_path+"fullimages_%d.npy"%id
# print(np.array(file_used).shape)
# imgs_to_process = np.load(file_used).astype(np.float64)
# plt.hist(imgs_to_process.flatten(), bins=50, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()

### Critiquing the Histogram
'''
The histogram suggests the following:
There is lots of air
There is some lung
There's an abundance of soft tissue, mostly muscle, liver, etc, but there's also some fat.
There is only a small bit of bone(seen as a tiny sliver of height between 700-3000)
...
'''

### Displaying an Image Stack, 3D voxels (Let's take a look at the actual images)
def viewer(samples, rows=5, cols=5, show_every=5):

    print("instance shape",samples.shape)

    for num, each_samples in enumerate(samples):
        fig = plt.figure()
        for ind, each_slices in enumerate(each_samples):

            #print("each_slices shape", each_slices.shape)
            ind+=1
            if ind/show_every > rows*cols:
                break
            if ind%show_every==0:
                y = fig.add_subplot(rows, cols, ind/show_every)  # it will be [3, 4] sub plot grid
                origin_shape = each_slices.shape
                y.imshow(np.reshape(each_slices, (origin_shape[0], origin_shape[1])))
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help="dicom directory you need to show")
    parser.add_argument('--output_path', type=str,
                        help="dir of numpy array that have dicoms pixels")
    parser.add_argument('--save_filename', type=str,
                        help="filename of numpy array that have dicoms pixels")
    parser.add_argument('--visualize', type=bool,
                        help="if you need to visualize")
    args = parser.parse_args()

    #data_path = "./test"  # argument1, required
    patients_dir = os.listdir(args.data_path)

    #output_path = "./dicom_viewer"  # argument2, optional

    #save_filename = '0' # argument3, required



    if os.path.isdir(args.output_path) is False:
        os.system('mkdir '+args.output_path)
    if os.path.isfile(os.path.join(args.output_path,args.save_filename)) is False:
        slices = []
        for each_dir in patients_dir:
            try:
                patient = load_scan(os.path.join(args.data_path, each_dir))
                slice = get_pixels_hu(patient)
                slices.append(slice)
            except NotADirectoryError as nade:
                print(nade)
                pass
        np.save(os.path.join(args.output_path,args.save_filename), slices)


    imgs_to_process = np.load(os.path.join(args.output_path, args.save_filename), encoding="latin1")
    print("dimension size of saved file",imgs_to_process.shape)

    if args.visualize is True:
        for ind in range(len(imgs_to_process)):
            if ind is 0:
                continue
            viewer(imgs_to_process[ind])

    from collections import Counter
    dimen_list = []
    for each_img in imgs_to_process:
        dimen_list.append(each_img.shape[0])
    print(Counter(dimen_list))



if __name__=="__main__":
    main()

'''
usage:
python dicom_viewer.py --data_path data/CE --output_path ./dicom_viewer --save_filename neurology.npy
python dicom_viewer.py --data_path data/LAA --output_path ./dicom_viewer --save_filename neurology.npy --visualize True

'''





