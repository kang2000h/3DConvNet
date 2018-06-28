import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--objective_path', type=str, help="objective_path having npyfile to show")
parser.add_argument('--filename', type=str, help="filename to show")
parser.add_argument('--label_to_show', type=int, help="label to show from the file")
parser.add_argument('--save_path', type=str, help="filename to save")

parser.add_argument('--rows', type=int, help="rows to show")
parser.add_argument('--cols', type=int, help="cols to show")
parser.add_argument('--step', type=int, help="step to pick up")



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
            y.imshow(np.reshape(each_slices, (origin_shape[0], origin_shape[1])))
    if save_path is not None:
        plt.savefig(save_path)
    else :
        plt.show()


def main():
    args = parser.parse_args()
    npy_voxels = np.load(os.path.join(args.objective_path, args.filename))

    label_voxels = np.array(npy_voxels[args.label_to_show])

    for ind, p in enumerate(label_voxels):
        print(str(ind)+" th on "+str(args.label_to_show)+"label")
        viewer(p, save_path=os.path.join(args.save_path, str(ind)+".png"), rows=args.rows, cols=args.cols, show_every=args.step)


if __name__ == "__main__":
    main()


