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

For the purposes of this project, we limit our scope and define a filter as a pair $(f, g)$ where $f: \mathbb{R}^3 \rightarrow \mathbb{R}^3$ is a function that maps every individual color (consisting of 3 channels each with a real value in the $[0, 1]$ range) to some color in the same range, and $g \in \mathbb{R}^{3 \times 3}$ is a convolution kernel that can be used for blurring and sharpening among other effects. We assume that a filter is applied first by passing each pixel of an image through $f$, and then convolving the image with $g$, extending the edges by repeating the last row and column of pixels as to preserve the shape of the image.

While many commercial filters may also contain additional effects such as borders and vignettes, filters are mostly characterized by how they shift the color curves globally and their blur/sharpen/emboss effects. Therefore, for the scope of this project, we choose filters which does not have these additional effect.

Though our work relates to many other fields of computer vision, such as image denoising and brightening images \cite{Dark}, not much work directly focuses on end to end filter identification and inversion. One publication that we found \cite{IEEE_Inversion} for identification depends heavily on prior knowledge of the camera demosaicing algorithm which is not always readily available. We thus chose to develop our own identification system.

In many of these settings such as image denoising or brightening, the modifications applied to the image (noise, etc.) are either consistent across the dataset or is known a priori. Our task is different from these previous work as our filter functions are unknown, but we have examples of unfiltered \& filtered images. Therefore, we decompose this task of filter inversion into two separate tasks, one is filter identification given an input image and the other is filter inversion given a known filter. Filter identification for an image is a classification task while filter inversion is a regression task estimating the filter inverses.

The problem of filter identification mirrors closely that of identification of the source camera of a given image. To identify the source camera, one needs to exploit the noise profiles inherent to a camera and identify that noise profile in the given image. There have been several pieces of literature, especially in the field of digital forensics, that attempts to model sensor noise patterns explicitly and build correlations between noise patterns and camera source[^lucas] but have failed to achieve high accuracy. However, several several recent works have applied convolutional neural networks to the problem and achieved notable results[^obregon] [^huang] [^kuzin]. Because the problem space between source camera identification and filter identification is similar, we take the approach of these recent papers and apply it to the context of filter inversion.


[^lucas]:([Lukas et al., 2006] Lukas, J., Fridrich, J., and Goljan, M. (2006). Digital camera identification from sensor pattern noise. IEEE Transactions on Information Forensics and Security, 1(2):205–214) 

[^obregon]:Deep learning for source camera identification on mobile devices https://arxiv.org/abs/1710.01257

[^huang]:  Identification of the source camera of images based on convolutional neural network https://www.sciencedirect.com/science/article/pii/S1742287618302664#bib18

[^kuzin]:Camera Model Identification Using Convolutional Neural Networks https://arxiv.org/pdf/1810.02981.pdf

## Approach

Our approach splits the end-to-end task of filter inversion into two steps:
1. Generate a probability vector for possible filters applied to a given image. (Filter classification)
2. With the image and the probability vector as inputs, apply a learned inverse filter onto the image to recover the unfiltered image. (Filter inversion)

While there are infinitely many filters possible, popular social media platforms have a few pre-selected filters that are widely used. Therefore, we constrain the scope of our filter inversion by assuming input images were filtered at most once by a filter from a known set. To accurately model a real-world application, our list comprises of the following six popular Instagram filters:


![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 23"


\def\filterimagewidth{3cm}
\begin{figure}[H]
    \centering
    \subfloat[Original image]{\label{fig:original}\includegraphics[width=\filterimagewidth]{images/original.jpg}}
    \subfloat[Clarendon]{\label{fig:original}\includegraphics[width=\filterimagewidth]{images/clarendon.jpg}}
    \subfloat[Gingham]{\label{fig:original}\includegraphics[width=\filterimagewidth]{images/gingham.jpg}}
    \subfloat[Juno]{\label{fig:original}\includegraphics[width=\filterimagewidth]{images/juno.jpg}}\\
    \subfloat[Lark]{\label{fig:original}\includegraphics[width=\filterimagewidth]{images/lark.jpg}}
    \subfloat[Gotham]{\label{fig:original}\includegraphics[width=\filterimagewidth]{images/gotham.jpg}}
    \subfloat[Reyes]{\label{fig:original}\includegraphics[width=\filterimagewidth]{images/reyes.jpg}}
    \caption{Selected filters}
    \label{fig:filters}
\end{figure}

Given the scant amount of existing literature on the problem of filter identification aside from \cite{IEEE_Inversion}, there were no established processes for filtering large numbers of images using commercial filters. We were prompted to create our own image filtering pipeline. Since Instagram filters are not available outside of their platform, we imitated these filters by manually modifying each color curve. We referenced channel adjustment code from an online article \cite{Instafilters}, which uses \verb|numpy| functions, specifically \verb|linspace| and \verb|interp|, to modify the color curves of each specific channel. We obtained curve parameters for each filter from \cite{Instafilters_tutorial} and passed them onto the channel adjustment code to create an imitation of commercial filters. We then run each imitation filter over our library of unfiltered images to create our dataset.

\subsection*{Filter classification}
Our approach to filter classification takes in an input image and outputs a probability vector for the possible filters applied to the input image. We utilize a neural network model to generate this probability vector from features extracted from the input image.

For feature extraction, because color curves are a major component of many of the popular image filters, we decided to use color histograms to extract global color information from the image. Furthermore, because these color curve modifications are often applied independently in each RGB channel, we create separate color intensity histograms for each color channel and concatenate them together to generate the features for a given image.

Note that this low level data can be augmented with scene and object information, premised on the idea that color distributions are correlated to the subjects and the environment of the image. This augmentation has not yet been incorporated into the pipeline.

Due to the lack of neural network based approaches in the previous work done in this area, we had no intuition on the appropriate complexity required for our models. Therefore, we first experimented with the simplest models with one layer and few neurons, found it performed poorly, and gradually increased complexity until diminishing return on performance occurred.

We utilize Keras \cite{Keras} to create a sequential, feed-forward neural network with varying number of layers at different sizes with the ReLU activation function on the hidden layers. The network ends with a softmax layer to obtain a probability vector. We use the ReLU activation function because it has been consistently shown to provide good performance and training speed for neural networks \cite{ReLU}. We use a cross-entropy loss function and the Adam optimizer \cite{Adam} to train our neural network model.

% TODO(chunlok): Add this back if needed.
One problem we encountered was that because each image passes through 6 different filters and each of these images are in our dataset, we have to ensure that our model has not seen the images before to avoid memorizing previous image color distributions to obtain good results in the testing set. Therefore, we utilize a completely different set of base images for the training and testing set.

\begin{figure}[H]
    \centering
    \input{detection_nn.tex}
    \caption{Detection NN Architecture}
    \label{fig:detection_nn}
\end{figure}



More specifically, we follow the approach stated in the first paper which detailed 2 convolutional layers followed by two fully connected layers into the output layer. The final output layer is a softmax layer and outputs a probability vector of filter applied to the image. Our current network's input size is only 32x32x3 and are trained using images of these size. For images larger than the network input size of 32x32x3, we subdivide the source image into separate patches and perform voting based on the classification for each subpatch.

We implemented the neural network using Keras api.
Since the miniplaces dataset used contains only 128x128x3 images, we subdivide each image into 16 disjoint subpatches of 32x32x3 images and use them as training data.  

## Experiments/Results

## Qualitative Results
## Conclusion and Future Work
## References



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTY5MjUyMjAzMV19
-->