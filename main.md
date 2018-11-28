# Hi Team

## Abstract

Given the prevalence of photo editing and filtering on popular social media platforms such as Snapchat and Instagram, it is becoming more and more difficult for users to identify how the content that they consume everyday have been modified from their original form. Since filtering has become so common, the goal of this project is to give users a means to identify how photos on these platforms are edited and to present a plausible reconstruction of source content. 

We propose an end-to-end system that will take an image from the user, identify probabilities for which common image filters were applied to the image, and apply the most likely filter inverse. We present both the filter probabilities and the inverted image to the user. We utilize features in the form of color histograms and scene details to extract a sense of natural color distributions and use neural networks both to determine the probabilities for applied filters and to invert an image given its most likely filter.

From our previous experiments, we were able to improve our results on classify images by the filter applied to it (from our predefined set of six Instagram filters) while distinguishing it from natural (unfiltered) images with an accuracy of 60% to 98% depending on the characteristics of the filter such as the amount of deviation from the original image.
Let $E(I, I')$ be the average per-pixel mean of the sum absolute differences in intensity across all color channels of images $I$ and $I'$ (\textit{Equation 1}). For inverting images given a known filter, we were previously able to obtain a pseudo-inverse of the image with an average error $E$ of 1\% and our end-to-end system detected and inverted filters with an average error $E$ of 5.5\% between our output image and the original unfiltered version. In comparison, the baseline error $E$ between filtered and unfiltered images was found to be 8.4\%.

Previously, the low accuracies of our simple filter identification model, with an average accuracy of 78% and a lowest F1 score across all filters of 0.61, presented a bottleneck for the overall quality of our filter inverses.
Here, we follow a new approach in filter prediction using convolutional neural networks. It was able to achieve an average accuracy of 95% and a lowest F1 score across all categories improved to be 0.88.

## Teaser Figure
## Introduction
Filtered photos have become ubiquitous on social media platforms such as Snapchat, Instagram, Flickr, and more. For the casual eye, the subtlety of these filters can make it hard to distinguish between filtered and unfiltered images on social media, leading to a false perceptual model of how natural images look, skewing our expectations about reality. We hope this project will help bring more transparency into how images are often edited by identifying whether a common image filter have been applied to an image, and expose users to the natural state of these images. We believe that transparency in the image editing process is important in raising awareness about deliberate modifications to perceptions of reality, allowing content consumers to enjoy the edited content while being aware of their true nature.

Not to be confused with filters in the computer vision setting, which are often used to better extract information from an image, filters in the social media setting describe a predefined set of modifications to an image that attempts enhance its human visual appeal. Most commonly, these filters come in the form of color balance adjustments and can be represented as tweaks to the color curves of an RGB image. A color curve $f: [0, 255] \to [0, 255]$ is a continuous function that remaps the intensities in each color channel. Modification to the color curve allows the user to non-uniformly boost or decrease color intensities at varying ranges to create various effects such as increasing contrast or creating color shifts (e.g. \cref{fig:color_curve} demonstrates a boost of blues in shadows while decreasing blues in highlights). Some filters also include additional effects such as blurring/sharpening using convolution kernels, the addition of borders, and the application of vignette darkening at the edges.

image!!!!




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
eyJoaXN0b3J5IjpbLTU4NjY2ODg0Myw0OTM5Nzc4MjgsLTE4Nj
I4Njc1MzcsODIwMjIzMTM1LC0xOTY3MjY1MTI2LDE5MDM5MDk2
MDVdfQ==
-->