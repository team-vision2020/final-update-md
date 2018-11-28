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

Filtered photos have become ubiquitous on social media platforms such as Snapchat, Instagram, Flickr, and more. To the casual viewer, filters can make it hard to detect, their intentional subtlety making it hard to distinguish between filtered and unfiltered images on social media. This can lead to distorted perceptions of natural images, skewing expectations about natural appearances. We hope this project will help bring more transparency into how images are often edited by identifying whether a common image filter has been applied to an image, and expose users to the natural state of these images. We believe that transparency in the image editing process is important in raising awareness about deliberate modifications to perceptions of reality, and hope to allow viewers to consume edited content while being aware of their true nature.

Not to be confused with filters in the computer vision setting, generally used in image preprocessing, filters in the social media setting describe a predefined set of modifications to an image that attempts enhance its human visual appeal. Most commonly, these filters come in the form of color balance adjustments and can be represented as tweaks to the color curves of an RGB image. A color curve $f: [0, 255] \to [0, 255]$ is a continuous function that remaps the intensities in each color channel. Modification to the color curve allows the user to non-uniformly boost or decrease color intensities at varying ranges to create various effects such as increasing contrast or creating color shifts (e.g. <!-- TODO: Add reference to the color curve figure. --> demonstrates a boost of blues in shadows while decreasing blues in highlights). Some filters also include additional effects such as blurring/sharpening using convolution kernels, the addition of borders, and the application of vignette darkening at the edges.

<!---
TODO: Add missing image
-->

For the purposes of this project, we limit our scope and define a filter as a pair $(f, g)$ where $f: \mathbb{R}^3 \rightarrow \mathbb{R}^3$ is a function that maps every individual color (consisting of 3 channels each with a real value in the $[0, 1]$ range) to some color in the same range, and $g \in \mathbb{R}^{3 \times 3}$ is a convolution kernel that can be used for blurring and sharpening among other effects. We assume that a filter is applied first by passing each pixel of an image through $f$, and then convolving the image with $g$, extending the edges by repeating the last row and column of pixels as to preserve the shape of the image.

While many commercial filters may also contain additional effects such as borders and vignettes, filters are mostly characterized by how they shift the color curves globally and their blur/sharpen/emboss effects. Therefore, for the scope of this project, we choose filters which does not have these additional effect.

Though our work relates to many other fields of computer vision, such as image denoising and brightening imagesDark], not much work directly focuses on end to end filter identification and inversion. One publication that we found nversion for identification depends heavily on prior knowledge of the camera demosaicing algorithm which is not always readily available. We thus chose to develop our own identification system.

In many of these settings such as image denoising or brightening, the modifications applied to the image (noise, etc.) are either consistent across the dataset or is known a priori. Our task is different from these previous work as our filter functions are unknown, but we have examples of unfiltered \& filtered images. Therefore, we decompose this task of filter inversion into two separate tasks, one is filter identification given an input image and the other is filter inversion given a known filter. Filter identification for an image is a classification task while filter inversion is a regression task estimating the filter inverses.

The problem of filter identification mirrors closely that of identification of the source camera of a given image. To identify the source camera, one needs to exploit the noise profiles inherent to a camera and identify that noise profile in the given image, similar to the 'noise' within a filtered image. There have been several pieces of literature, especially in the field of digital forensics, that attempts to model sensor noise patterns explicitly and build correlations between noise patterns and camera source[^lucas] but have failed to achieve high accuracy. However, several several recent works have applied convolutional neural networks to the problem and achieved notable results[^obregon] [^huang] [^kuzin]. Because the problem space between source camera identification and filter identification is similar, we take the approach of these recent papers and apply it to the context of filter inversion.

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
<!---
TODO: Add missing image
-->

Given the scant amount of existing literature on the problem of filter identification aside from ie_[^ieee_inversion], there were no established processes for filtering large numbers of images using commercial filters. We were prompted to create our own image filtering pipeline. Since Instagram filters are not available outside of their platform, we imitated these filters by manually modifying each color curve. We referenced channel adjustment code from an online article [^Instafilters], which uses `numpy` functions, specifically `linspace` and `interp`, to modify the color curves of each specific channel. We obtained curve parameters for each filter from [^Instafilters_tutorial] and passed them onto the channel adjustment code to create an imitation of commercial filters. We then run each imitation filter over our library of unfiltered images to create our dataset.

### Filter classification
Our approach to filter classification takes in an input image and outputs a probability vector for the possible filters applied to the input image. We utilize a neural network model to generate this probability vector from the input image.

As convolutional neural network architecture was able to obtain good results in the problem of source camera identification, we follow the architecture detailed in the paper by D. Freire-Obregon[^obregon] and apply their technique to this problem space. 

We utilize Keras[^Keras] to create a convolutional neural network that takes in $32 \times 32 \times 3$ images, pass them through two convolutional layers, one max pool layer, and two fully connected layer before passing it through a softmax layer to output a probability vector of which filter the model predicts the image have been passed through. We use categorical cross-entropy loss function and the Adam optimizer [^Adam] to train our neural network model.

We compared the use of ReLU and leaky ReLU activation function in our network. We found that the use of ReLU often caused our network to not train at all and found that leaky ReLU  provides vastly superior results in our use case. We further determined the non-activation slope of leaky ReLU experimentally to be 0.3.

While many papers utilizes regularization and dropout layers to reduce overfitting in the training process, we have found no evidence of overfitting in our network due to our large number of training examples (over 900k images). Therefore, we did not use any regularization and dropout layers.

#### Network Architecture
<!---
TODO: Add missing image. Probably rip image from paper 2? No clue how they make CNN diagrams]
-->

Because the architecture presented in the paper treats only 32x32x3 images while our dataset and input consisted of larger images, we developed our own routines to divide our input image into separate 32x32x3 images. During prediction time, a voting scheme is used from the prediction output from each 32x32x3 subpatch to make the final decision. This makes our architecture adaptable to various image sizes and have experimentally be shown to further increase prediction accuracy.

One problem we encountered was that because each image passes through 6 different filters and each of these images are in our dataset, we have to ensure that our model has not seen the images before to avoid memorizing previous image color distributions to obtain good results in the testing set. Therefore, we utilize a completely different set of base images for the training, testing, and validation set.


### Filter Inversion


## Experiments/Results
We perform our experiments using 9000 $128\ \times\ 128 \times 3$ images from 10 different categories from the MiniPlaces dataset[^Places] passed through 6 different filters [^TODO: filters] to create a total dataset of 63000 images (including the original images.) We split these images into 89.55% training 0.45% validation and 10\% testing sets with assurance that each image and its derivatives all belong to a single set. Therefore, our training set consists of 56420 images, our validation set consisted of 280 images, and our testing set consisted of 6300 images.

$$ \text{9000 images} \times \text{6 filters} + \text{9000 original images} = \text{63000 training images}$$

$$ \text{63000 images} \times 0.8955 \approx 56420$$

$$ \text{63000 images} \times 0.45\approx 280$$

$$ \text{63000 images} \times 0.1 = 6300$$

For filter identification, because each image in the dataset is $128 \times 128 \times 3$, we further subdivide each image into 16 disjoint $32 \times 32 \times 3$ image for training, leading to a total of 902720 training images and 4480 validation images. Each image also have an associated 7-dimensional one-hot output vector that indicates the ground truth for which filter has been applied to the image.

We adopted the approach in [^obregon] and found that slight tweaks in its hyperparameters was sufficient to provide strong results. We test greater modifications to the architecture such as adding additional convolution layers and tweaking number of neurons in the fully connected layers but found decreasing model complexity generally decrease accuracy and increasing complexity did not lead to meaningful increase in accuracy. One specific thing we did tune was  the negative slope coefficient on the leaky ReLU. We found a sweet spot of 0.3 to that allowed both fast convergence and robustness to the variation of hyperparameters.

We trained in rounds of 5 epochs with various batch sizes until validation accuracy started to decrease due to overfitting. Our model is trained with a total of 3 rounds of 5 epochs with batch sizes of 256, 1024, and 4096 on a Nvidia GTX970m GPU using Keras.

We evaluated how many epochs to train using overall accuracy on the validation set for each individual $32 \times 32 \times 3$  image to prevent overfitting. 

For final model evaluation, we evaluated the overall accuracy, precision, recall, and F1-score for each filter category using the testing set. We also evaluated our model based on both individual $32 \times 32 \times 3$ image labeling and prediction using a majority vote voting scheme.

We also compared this approach to the results of our previous approach, where we extracted RGB color histograms as features to an image and passing that to a feed-forward neural network model for classification.

Baseline accuracy with random decision: $0.143$
Average accuracy from color histogram features feeding into a neural network model:  $0.783$
Average accuracy from feeding additional context information to neural network model:  $0.783$

**Average accuracy of our convolutional neural network model: $0.955$**

Result from Individual Image classification:


| Metric \ Filter| Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |



<!--
While we experimented with greyscale color histogram at first for its simplicity, the important role of color in filter identification pushed us towards our current feature extraction method. And because filters often modify color curves within the RGB space, we decided to extract three separate color intensity histogram in the RGB channel and concatenate them together as our image feature. We use 255 bins per color channel, which were represented as floats in the range [0, 1]. No meaningful performance gain was observed when increasing the number of bins past 255. Our neural network hyperparameters were tuned through manual search by starting with a simple model and increasing model complexity until no apparent improvements was noticed. 

Given the simplicity of features we extract, we expected a simple model to provide comparative performance in the filter identification task. However, after experimentation, adding multiple layers to our model increased our model accuracy by upwards of 8\%. It is possible that applying a filter might affect an image's color histogram in more complex ways than we assumed and increasing the number of layers enabled the network to better understand the effect of filter application on an image's color histogram.

Our final neural network contains four size 32 layers, one size 16 layer, two size 8 layers, and an output softmax layer with 7 nodes, in that order. The network is fully connected and every layer except the last uses \textit{ReLU} activation. The softmax layer produces a probability vector for the predicted filter. An input feature vector for this model contains $255 \times 3 = 765$ features. The main hyperparameter for future experimentation is the number of bins used here, or in general, what we choose to feed into our network. When stepping through increasing bin count, we found performance plateaued at 255 bins.
We trained our neural network classifier for a total of 100 epochs with a batch size of 128 using the Adam optimizer \cite{Adam} and the cross-entropy loss function.

One improvement to note is that we did not utilize validation set (which we should have) during our training process so we do not have an exact measure of overfitting. However, we evaluated our model performance different number of epochs and there were no noticeable increase accuracy from stopping before or going past 100 epochs.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/confusionMatrix.jpg}
    \caption{Confusion matrix for detection NN}
    \label{fig:confusion_matrix}
\end{figure} 


Our initial approach evaluated our model based on the overall accuracy in the prediction (filter with maximum probability from our probability vector). However, we noticed visually subtler filters naturally had lower identification rates, so we refined our statistics to distinguish success for different filters. We thus also looked at the precision, recall, F1 score, and the confusion matrix to evaluate model performance on individual filters. For comparison, the baseline accuracy of a random classifier is 0.143. -->

## Qualitative Results

## Conclusion and Future Work

We initially considered a systematic approach using nearest neighbors in a large corpus of knowledge about color distributions of scenes and detected objects, but decided to drop this alternative as the neural nets quickly pulled ahead in performance. Adding scene information did further improve performance, however, meaning that as suspected the network did gain knowledge of color distributions for different scenes. Initial plans were to construct a voting system over detected object masks, thereby exploiting color distributions of common objects. Due to the intractability of MaskRCNN with such a large dataset given our limited resources,  we instead used voting over fixed size patches in our image. Similar to how scene information was easily incorporated into the neural network approach, future work could force attention on objects in the scene by adding variable length feature composed of detected objects. Another alternative would be to cluster detectable objects by average color distributions and then create a 'bag of objects' fixed length feature that could be added to our input. 

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

[^Places]: B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba, "Places: A 10 million Image Database for Scene Recognition", IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.

[^Dark]: C. Chen, C. Chen, J. Xu, and V. Koltun, "Learning to See in the Dark", CVPR, 2018.

[^LeakyReLU]:Bing Xu and Naiyan Wang and Tianqi Chen and Mu Li, "  
Empirical Evaluation of Rectified Activations in Convolutional Network", [arXiv](https://arxiv.org/abs/1505.00853), 2015.

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTkzOTE2NzUyMywtNDMwNjQ1MjUyLC0yMD
UyNjA4NDIsLTI1OTAwNzU3MywxNzkyMjgxMTUsMTE0MzU3NDU5
LC0zMzIyOTkyMDYsNzg1Njc1MjgyLDE2OTQ2MjcwNTcsLTE5Nj
A2NzQ1LC0xNzk5MTE3Njg1LC05NjgyMjkwNjQsNTAyNDUyOTA3
LDIwMDMyNDE2OTcsMTMwMjYwODExMCw3ODA4ODU1MDMsLTIxMT
c3NDY5ODUsMjAyOTc0MTU4MSw2Nzg0MTMwNzMsMTUzMjgxOTkw
XX0=
-->