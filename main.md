# Hi Team

## Abstract
## Teaser Figure
## Introduction
The problem of filter identification mirrors closely that identification of camera source based on camera noise and there are several piece of literature regarding this topic. While there have been several paper on this topic previously using traditional expert feature based techniques [need references], we particularly follow convolutional neural network based approaches that have shown good results in the past two years, specifically

##### Deep learning for source camera identification on mobile devices
https://arxiv.org/abs/1710.01257

#####  Identification of the source camera of images based on convolutional neural network
https://www.sciencedirect.com/science/article/pii/S1742287618302664#bib18

##### Camera Model Identification Using Convolutional Neural Networks
https://arxiv.org/pdf/1810.02981.pdf

## Approach
More specifically, we follow the approach stated in the first paper which detailed 2 convolutional layers followed by two fully connected layers into the output layer. The final output layer is a softmax layer and outputs a probability vector of filter applied to the image. Our current network's input size is only 32x32x3 and are trained using images of these size. For images larger than the network input size of 32x32x3, we subdivide the source image into separate patches and perform voting based on the classification for each subpatch.

We implemented the neural network using Keras api.
Since the miniplaces dataset used contains only 128x128x3 images, we subdivide each image into 16 disjoint subpatches of 32x32x3 images and use them as training data.  

## Experiments/Results

## Qualitative Results
## Conclusion and Future Work
## References



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4NjI4Njc1MzcsODIwMjIzMTM1LC0xOT
Y3MjY1MTI2LDE5MDM5MDk2MDVdfQ==
-->