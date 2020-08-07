# 3D-RCNN
This repository contains an implementation of "3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare, CVPR 2018". 
Please note that this is not the official implementation.
In progress...

## Installation
Install Anaconda3

Run the following commands to create conda environment and install all dependencies:
```console
username@PC:~$ conda env create -f environment.yml
username@PC:~$ conda activate 3drcnn
```

## Run
main.py contains all the steps required to run the network.

model.py includes all the models (including shapenet, posenet, and the backbone).

data_loader.py contains the implementation for all the datasets used in this work.

render.py contains the code for renderer in Pytorch3d.

config.yaml contains all the configurations used by main.py, model.py, and data_loader.py, and render.py

## Datasets
The ShapeNet and PoseNet are pretrained on ShapeNet dataset which will be uploaded here. The network is trained end to end on KITTI dataset afterwards for instance level 3D scene understanding.

The code to generate multiple objects per image for training of ShapeNet and PoseNet will be available soon.

## Quantitative Results
In progress...

## Qualitative Results
In progress...
