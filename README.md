# 3D Convolutional Neural Network

I studied 3D CNN and other hyperparameters to improve the performance of one's models. 
So I just implemented 3D CNN framework for my studies, I refered to many of programs from github(digits, [awp4211's lung repo](https://github.com/awp4211/lung)) to achieve better performance.
I just wanted to create a framework for me having divided some module to concentrate on each train/val/test phase respectively.

I'll rearrange this repo in a calm and orderly way

## Usage

### dicom files preprocessing & converting

dcm2nii.py : convert dcm file into nifti file.

`python dcm2nii.py --data_dir [dcm's upper-upper dir] --output_dir [savepath]`

dicom_loader.py / nifti_loader.py : make dicom/nifti file into npy file to separate source file and data processing pipeline.

dicom_loader2.py / nifti_loader2.py : running like above, but internally create and use a process handling p_id.

npy_holdout.py : make npy file into separated npy files with fixed data structure.

npy_holdout2.py : running like above, but internally create and use a process handling p_id. 

main.py : manage 3DConvNet module according to intended function.

### train phase
`python main.py --mode=train --data_path=./data/kaggle_data.npy --model_dir=./tmp/checkpoint --learning_rate=0.00001 --epoch=200`

`python main.py --mode=train --data_path=./data/neurology-250-250-20.npy --model_dir=./tmp/checkpoint --learning_rate=0.00001 --epoch=200`

### val phase
`python main.py --mode=val --tv_type=cnn_ho --data_path=./data/neurology-250-250-20.npy --model_dir=./tmp/checkpoint --learning_rate=0.00001 --epoch=200`


### test phase
`python main.py --mode=test --data_path=./data/neurology-250-250-20.npy --model_dir=./tmp/checkpoint`

