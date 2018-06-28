from nibabel import load as nib_load
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def viewer(samples, rows=5, cols=5, show_every=5):

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
            y.imshow(np.reshape(each_slices, (origin_shape[0], origin_shape[1])))
    plt.show()

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

objective_path = "/home/galaxy2/database/home/galaxy2/dataset/npy_data"
filename = "norm_amyloid_68_79_95.npy"
voxel = np.load(os.path.join(objective_path, filename))
print("#########3", np.array(voxel[0]).shape)
print("#########3", np.array(voxel[1]).shape)
print("#########3", np.array(voxel[2]).shape)


mask_path = "/home/galaxy2/database/home/galaxy2/dataset/mask/threshold_0.5/revised"
mask_name = "brainmask_grey_resize_79_95_68.nii"
mask = nib_load(os.path.join(mask_path, mask_name))
mask = np.array(mask.get_data())
mask_s = mask.shape


resized_mask = np.resize(mask, (mask_s[0], mask_s[1], mask_s[2]))
#resized_mask = resize_3D(mask, (mask_s[0], mask_s[1], mask_s[2]))
resized_mask = np.transpose(resized_mask, [2,0,1])

#print(resized_voxel.shape)


# for ax in mask_voxel:
#     plt.imshow(ax)
#     plt.show()


#masked_voxel = p_voxel*resized_voxel
# for ax in masked_voxel:
#     plt.imshow(ax)
#     plt.show()


# print("mask", mask_voxel.shape)
resized_mask = np.expand_dims(resized_mask, axis=len(np.array(resized_mask).shape))
print("expanded mask", resized_mask.shape)



c3_list = []
c1_list = []
c2_list = []




for c3 in voxel[0]:
    c3_list.append(np.array(c3*resized_mask))
for c1 in voxel[1]:
    c1_list.append(np.array(c1 * resized_mask))
for c2 in voxel[2]:
    c2_list.append(np.array(c2 * resized_mask))


c1_list = np.array(c1_list)
c2_list = np.array(c2_list)
c3_list = np.array(c3_list)


dest_filename = "grey_masked_amyloid_68_79_95.npy"
#np.save(dest_filename, np.array([c1_list, c2_list, c3_list]))

#print("R", np.array(c1_list[0].shape))

for ind in range(0, len(c1_list), 50):
    print(str(ind)+"th voxel")
    viewer(c1_list[ind], rows=5, cols=6, show_every=5)
#viewer(mask_voxel, rows=5, cols=6, show_every=5)
#viewer(resized_mask, rows=5, cols=6, show_every=5)