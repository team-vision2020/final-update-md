# Automatic Image Filter Identification
**Mert Dumenci, Cem Gokmen, Chunlok Lo, Joel Ye**
CS4476 Term Project, Fall 2018
Georgia Tech

## Abstract

Given the prevalence of photo editing and filtering on popular social media platforms such as Snapchat and Instagram, it is becoming more and more difficult for users to identify how the content that they consume everyday have been modified from their original form. Since filtering has become so common, the goal of this project is to give users a means to identify how photos on these platforms are edited and to present a plausible reconstruction of source content. 

We propose an end-to-end system that will take an image from the user, identify probabilities for which common image filters were applied to the image, and apply the most likely filter inverse. We present both the filter probabilities and the inverted image to the user. We utilize features in the form of color histograms and scene details to extract a sense of natural color distributions and use neural networks both to determine the probabilities for applied filters and to invert an image given its most likely filter.

From our previous experiments, we were able to improve our results on classify images by the filter applied to it (from our predefined set of six Instagram filters) while distinguishing it from natural (unfiltered) images with an accuracy of 60% to 98% depending on the characteristics of the filter such as the amount of deviation from the original image.
Let $E(I, I')$ be the average per-pixel mean of the sum absolute differences in intensity across all color channels of images $I$ and $I'$ (_Equation 1) For inverting images given a known filter, we were previously able to obtain a pseudo-inverse of the image with an average error $E$ of 1\% and our end-to-end system detected and inverted filters with an average error $E$ of 5.5\% between our output image and the original unfiltered version. In comparison, the baseline error $E$ between filtered and unfiltered images was found to be 8.4\%.

Previously, the low accuracies of our simple filter identification model, with an average accuracy of 78% and a lowest F1 score across all filters of 0.61, presented a bottleneck for the overall quality of our filter inverses.
Here, we follow a new approach in filter prediction using convolutional neural networks. It was able to achieve an average accuracy of 95% and a lowest F1 score across all categories improved to be 0.88.

## Teaser Figure

<!---
TODO: Add missing image
-->

## Introduction

Filtered photos have become ubiquitous on social media platforms such as Snapchat, Instagram, Flickr, and more. For the casual eye, the subtlety of these filters can make it hard to distinguish between filtered and unfiltered images on social media, leading to a false perceptual model of how natural images look, skewing our expectations about reality. We hope this project will help bring more transparency into how images are often edited by identifying whether a common image filter have been applied to an image, and expose users to the natural state of these images. We believe that transparency in the image editing process is important in raising awareness about deliberate modifications to perceptions of reality, allowing content consumers to enjoy the edited content while being aware of their true nature.

Not to be confused with filters in the computer vision setting, which are often used to better extract information from an image, filters in the social media setting describe a predefined set of modifications to an image that attempts enhance its human visual appeal. Most commonly, these filters come in the form of color balance adjustments and can be represented as tweaks to the color curves of an RGB image. A color curve $f: [0, 255] \to [0, 255]$ is a continuous function that remaps the intensities in each color channel. Modification to the color curve allows the user to non-uniformly boost or decrease color intensities at varying ranges to create various effects such as increasing contrast or creating color shifts (e.g. <!-- TODO: Add reference to the color curve figure. --> demonstrates a boost of blues in shadows while decreasing blues in highlights). Some filters also include additional effects such as blurring/sharpening using convolution kernels, the addition of borders, and the application of vignette darkening at the edges.

<!---
TODO: Add missing image
-->

For the purposes of this project, we limit our scope and define a filter as a pair $(f, g)$ where $f: \mathbb{R}^3 \rightarrow \mathbb{R}^3$ is a function that maps every individual color (consisting of 3 channels each with a real value in the $[0, 1]$ range) to some color in the same range, and $g \in \mathbb{R}^{3 \times 3}$ is a convolution kernel that can be used for blurring and sharpening among other effects. We assume that a filter is applied first by passing each pixel of an image through $f$, and then convolving the image with $g$, extending the edges by repeating the last row and column of pixels as to preserve the shape of the image.

While many commercial filters may also contain additional effects such as borders and vignettes, filters are mostly characterized by how they shift the color curves globally and their blur/sharpen/emboss effects. Therefore, for the scope of this project, we choose filters which does not have these additional effect.

Though our work relates to many other fields of computer vision, such as image denoising and brightening imagesDark], not much work directly focuses on end to end filter identification and inversion. One publication that we found nversion for identification depends heavily on prior knowledge of the camera demosaicing algorithm which is not always readily available. We thus chose to develop our own identification system.

In many of these settings such as image denoising or brightening, the modifications applied to the image (noise, etc.) are either consistent across the dataset or is known a priori. Our task is different from these previous work as our filter functions are unknown, but we have examples of unfiltered \& filtered images. Therefore, we decompose this task of filter inversion into two separate tasks, one is filter identification given an input image and the other is filter inversion given a known filter. Filter identification for an image is a classification task while filter inversion is a regression task estimating the filter inverses.

The problem of filter identification mirrors closely that of identification of the source camera of a given image. To identify the source camera, one needs to exploit the noise profiles inherent to a camera and identify that noise profile in the given image. There have been several pieces of literature, especially in the field of digital forensics, that attempts to model sensor noise patterns explicitly and build correlations between noise patterns and camera source[^lucas] but have failed to achieve high accuracy. However, several several recent works have applied convolutional neural networks to the problem and achieved notable results[^obregon] [^huang] [^kuzin]. Because the problem space between source camera identification and filter identification is similar, we take the approach of these recent papers and apply it to the context of filter inversion.

## Approach

Our approach splits the end-to-end task of filter inversion into two steps:

* Generate a probability vector for possible filters applied to a given image. (Filter classification)
* With the image and the probability vector as inputs, apply a learned inverse filter onto the image to recover the unfiltered image. (Filter inversion)

While there are infinitely many filters possible, popular social media platforms have a few pre-selected filters that are widely used. Therefore, we constrain the scope of our filter inversion by assuming input images were filtered at most once by a filter from a known set. To accurately model a real-world application, our list comprises of the following six popular Instagram filters:

1. Clarendon
2. Gingham
3. Juno
4. Lark
5. Gotham
6. Reyes
7. 
<!---
TODO: Add missing image
-->

Given the scant amount of existing literature on the problem of filter identification aside from ie_[^ieee_inversion], there were no established processes for filtering large numbers of images using commercial filters. We were prompted to create our own image filtering pipeline. Since Instagram filters are not available outside of their platform, we imitated these filters by manually modifying each color curve. We referenced channel adjustment code from an online article [^Instafilters], which uses `numpy` functions, specifically `linspace` and `interp`, to modify the color curves of each specific channel. We obtained curve parameters for each filter from [^Instafilters_tutorial] and passed them onto the channel adjustment code to create an imitation of commercial filters. We then run each imitation filter over our library of unfiltered images to create our dataset.

## Filter classification
Our approach to filter classification takes in an input image and outputs a probability vector for the possible filters applied to the input image. We utilize a neural network model to generate this probability vector from features extracted from the input image.

As convolutional neural network architecture was able to obtain good results in the problem of source camera identification, we follow the architecture detailed in the paper by D. Freire-Obregon[^obregon] and apply their technique to this problem space. 

We utilize Keras[^Keras] toWe create a convolutional neural network that takes in $32 \times 32 \times 3$ images, pass them through two convolutional layers, one max pool layer, and two fully connected layer before passing it through a softmax layer to output a probability vector of which filter the model predicts the image have been passed through.



Similar to in the paper, compared the use of ReLU and leaky ReLU activation function in our network and found that leaky ReLU provided vastly superior results. We further determined the non-activation slope of leaky ReLU experimentally to be 0.3

o eur etraciecures ae o otan o eults i te olo soramera infation fo the artere eae thee  reionreon ad ply i echne  t prole sae crae a conotona eura net that tae in   ima, ps the tht cooutiona lae oe a oo ae an o y connete laer eore pas i throu  ot ae  otut  proility eor of w fiter the models the ae ae ee pased throu. We use categorical cross-entropy loss function and the Adam optimizer [^Adam] to train our neural network model.

 further we compared the use of ReLU and leaky ReLU activation function in our network. We found that the use of ReLU often caused our network to not train at all and found that leaky ReLU provide vastly superior results. We further determined the non-activation slope of leaky ReLU experimentally to be 0.3




One problem we encountered was that because each image passes through 6 different filters and each of these images are in our dataset, we have to ensure that our model has not seen the images before to avoid memorizing previous image color distributions to obtain good results in the testing set. Therefore, we utilize a completely different set of base images for the training and testing set.

<!---
TODO: Add missing image
-->

More specifically, we follow the approach stated in the first paper which detailed 2 convolutional layers followed by two fully connected layers into the output layer. The final output layer is a softmax layer and outputs a probability vector of filter applied to the image. Our current network's input size is only $32 \times 32 \times 3$ and are trained using images of these size. For images larger than the network input size of $32 \times 32 \times 3$, we subdivide the source image into separate patches and perform voting based on the classification for each subpatch.

We implemented the neural network using the Keras[^Keras] API, and the TensorFlow backend.

Since the miniplaces dataset used contains only $128 \times 128 \times 3$ images, we subdivide each image into 16 disjoint subpatches of $32 \times 32 \times 3$ images and use them as training data.  

## Experiments/Results

## Qualitative Results

## Conclusion and Future Work

## References

[^ieee_inversion]: C. Chen and M. C. Stamm, “Image filter identification using demosaicing residual features,” 2017 IEEE International Conference on Image Processing (ICIP), Beijing, 2017, pp. 4103-4107.

[^Keras]: F. Chollet and others, ''Keras'', GitHub, 2015.

[^Adam]: D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", 3rd International Conference for Learning Representations, San Diego, 2015.

[^lucas]: J. Lukas,  J. Fridrich, and M. Goljan, ''Digital camera identification from sensor pattern noise.", IEEE Transactions on Information Forensics and Security, 2006.

[^obregon]: D. Freire-Obregon, F. Narducci, S. Barra, and M. Castrillon-Santana. "Deep learning for source camera identification on mobile devices", [arXiv](https://arxiv.org/abs/1710.01257), 2017.

[^huang]: N. Huang, J. He, X. Xuan, G. Liu, and C. Chang. "<!-- [NaHuanga](https://www.sciencedirect.com/science/article/pii/S1742287618302664#!)[JingshaHea](https://www.sciencedirect.com/science/article/pii/S1742287618302664#!)[NafeiZhua](https://www.sciencedirect.com/science/article/pii/S1742287618302664#!)[XinggangXuana](https://www.sciencedirect.com/science/article/pii/S1742287618302664#!)[GongzhengLiua](https://www.sciencedirect.com/science/article/pii/S1742287618302664#!)[ChengyueChangb](https://www.sciencedirect.com/science/article/pii/S1742287618302664#!) -->

[^huang]: N. Huang, J. Hea, XIdentification of the source camera of images based on convolutional neural network", Digital Investigation, 20 https://www.sciencedirect.com/science/article/pii/S1742287618302664#bib18.

[^kuzin]: A. Kuzin, A. Fattakhov, I. Kibardin, V. Iglovikov, and R. Dautov. "Camera Model Identification Using Convolutional Neural Networks", [arXiv]( https://arxiv.org/abspdf/1810.02981), 2018..pdf

[^Instafilters]: M. Pratusevich, "Instagram Filters in 15 Lines of Python", Practice Python, 2016. _Retrieved from [URL](https://www.practicepython.org/blog/2016/12/20/instagram-filters-python.html)_.

[^Instafilters_tutorial]: GraphixTV, "Instagram Filter Effects Tutorials", YouTube, 2017. _Retrieved from [URL](https://www.youtube.com/playlist?list=PLESCEzav4FyfUIi0RHMkNbQI-9JVr4kJc)_

[^MaskRCNN]: H. Kaiming, G. Gkioxari, P. Dollar and R. B. Girshick, "Mask R-CNN", Facebook AI Research, 2018.

[^Places]: B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba, "Places: A 10 million Image Database for Scene Recognition", IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.

[^Dark]: C. Chen, C. Chen, J. Xu, and V. Koltun, "Learning to See in the Dark", CVPR, 2018.

[^ReLU]: R. K. Srivastava, J. Masci, F. Gomez and J. Schmidhuber, "Understanding Locally Competitive Networks", ICLR, 2015.

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg3Nzg1ODg1NiwtMTc5OTExNzY4NSwtOT
Y4MjI5MDY0LDUwMjQ1MjkwNywyMDAzMjQxNjk3LDEzMDI2MDgx
MTAsNzgwODg1NTAzLC0yMTE3NzQ2OTg1LDIwMjk3NDE1ODEsNj
c4NDEzMDczLDE1MzI4MTk5MCwxNzQxNjA5MDYyLC02OTI1MjIw
MzEsOTIyOTY4NTcsLTk2MDE0NzQxNiw1MDA3OTg5MTMsLTE2Nj
E1Njc2OTYsNDkzOTc3ODI4LC0xODYyODY3NTM3LDgyMDIyMzEz
NV19
-->