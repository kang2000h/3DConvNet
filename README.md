# 3D Convolutional Neural Network

I studied 3D CNN and other hyperparameter to improve the performance of one's model. 
So I just implemented 3D CNN Framework for my study, I refered to many of programs from github(digits, [awp4211's lung repo](https://github.com/awp4211/lung)) to achieve better performance.
I just wanted to create a framework for me having divided some module to concentrate each train/val/test phase respectively.

I'll rearrange this repo in a calm and orderly way

## Usage

`python main.py --mode=train --data_path=./data/kaggle_data.npy --model_dir=./tmp/checkpoint --learning_rate=0.00001 --epoch=200`

`python main.py --mode=train --data_path=./data/neurology-250-250-20.npy --model_dir=./tmp/checkpoint --learning_rate=0.00001 --epoch=200`
