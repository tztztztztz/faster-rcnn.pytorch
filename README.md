# Faster-RCNN Pytorch Implementaton

This is a simple implementation of [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497).  I mainly referred to two repositories below.
- [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)
- [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/c7df5c92f42b41b7151156c04a6e26b1f8911516)

## Prerequisites

- python 3.5.x
- pytorch 0.4.1
- tensorboardX
- opencv3
- pillow
- easydict
- matplotlib

## Purpose

- [x] Training and testing on VOC
- [x] ResNet-50
- [x] ResNet-101

## Main Results

voc07+12trainval/voc07test

|original/res101|this/res101 | this/res50|
| :--: | :--: | :--: |
|76.4|76.9|76.0|

Running time: ~100ms(10FPS) on a GTX 1080

## Visualization

You can inspect the accuray of both foreground and background samples in RPN and RCNN 

<div style="color:#0000FF" align="center">
<img src="images/rpn_log.png" width="100%"/>
<img src="images/rcnn_log.png" width="100%"/> 
</div>

## Detection Results


<div style="color:#0000FF" align="center">
<img src="images/det_1.png" width="350px"/>
<img src="images/det_2.png" width="350px"/>
</div>


## Preparation

1. First clone the code

        git clone https://github.com/tztztztztz/faster-rcnn.pytorch.git
    
2. Install dependencies
	
        cd $PROJECT
        pip install -r requirements.txt
	
3. Compile `roi_pooling layer` and `gpu_nms`
    
        cd $PROJECT/faster_rcnn
        sh make.sh

## Training on PASCAL VOC

### Prepare the data

Please follow the instructions of this [repository](https://github.com/tztztztztz/yolov2.pytorch.git) to prepare the data

### Download the pretrained model on ImageNet

Get the model [ResNet-50](https://drive.google.com/open?id=0B7fNdx_jAqhtbllXbWxMVEdZclE) [ResNet-101](https://drive.google.com/open?id=0B7fNdx_jAqhtbllXbWxMVEdZclE), and put it at `$PROJECT/data/pretrained` folder. See more detail at [ruotian/pytorch-resent](https://github.com/ruotianluo/pytorch-resnet)

### Train
    python train.py --cuda
    
## Evaluation

    python test.py --cuda
    
you can check the detection results with command below

    python test.py --cuda --vis
    





    

