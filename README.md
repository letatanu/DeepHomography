# Deep Image Homography Estimation
This project is the unoffical implementation of the paper [Deep Image Homography Estimation](https://arxiv.org/abs/1606.03798).
A homography is a mapping from a projective space (image) P to Q. From this network, it will estimate 
a `4-point homography parameterization which maps the four corners from one image into the second image`.

# Dataset 

I used MS-COCO dataset as it described in the paper. You can download it from [here](https://cocodataset.org/#download)
There are 118287 images in the train set, and 40670 in the test set. 
## Getting the dataset


## Pre-processing
There are two ways to prepare the dataset. 
1. Pre-processing 
2. 