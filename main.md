# Hi Team

## Abstract

Given the prevalence of photo editing and filtering on popular social media platforms such as Snapchat and Instagram, it is becoming more and more difficult for users to identify how the content that they consume everyday have been modified from their original form. Since filtering has become so common, the goal of this project is to give users a means to identify how photos on these platforms are edited and to present a plausible reconstruction of source content. 

We propose an end-to-end system that will take an image from the user, identify probabilities for which common image filters were applied to the image, and apply the most likely filter inverse. We present both the filter probabilities and the inverted image to the user. We utilize features in the form of color histograms and scene details to extract a sense of natural color distributions and use neural networks both to determine the probabilities for applied filters and to invert an image given its most likely filter.

From our previous experiments, we were able to improve our results on classify images by the filter applied to it (from our predefined set of six Instagram filters) while distinguishing it from natural (unfiltered) images with an accuracy of 60% to 98% depending on the characteristics of the filter such as the amount of deviation from the original image.
Let $E(I, I')$ be the average per-pixel mean of the sum absolute differences in intensity across all color channels of images $I$ and $I'$ (\textit{Equation 1}). For inverting images given a known filter, we were previously able to obtain a pseudo-inverse of the image with an average error $E$ of 1\% and our end-to-end system detected and inverted filters with an average error $E$ of 5.5\% between our output image and the original unfiltered version. In comparison, the baseline error $E$ between filtered and unfiltered images was found to be 8.4\%.

The low accuracies of our simple filter identification model previously presented a bottleneck for the overall quality of our filter inverses. 
We have improved our results in filter prediction using a convolutional neural network from an average of 78% previously to an average of 95% accuracy. Our lowest F1 score across all categories improved from 0.61 to 0.88.

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
eyJoaXN0b3J5IjpbLTE1MDg0Mzg2OTYsNDkzOTc3ODI4LC0xOD
YyODY3NTM3LDgyMDIyMzEzNSwtMTk2NzI2NTEyNiwxOTAzOTA5
NjA1XX0=
-->