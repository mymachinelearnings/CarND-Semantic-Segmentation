## Self-Driving Car Engineer Nanodegree Program
## Term 3 Project 1
### Path Planning Project

---

### Writeup Template
This document explains the Semantic Segmentation project that involves building a Fully Convolutional Network(FCN) and discusses the architecture. The outcome of this project will be a pixel wise labelling of the input image

---
### Rubric Points
Here are the [rubric points](https://review.udacity.com/#!/rubrics/989/view) for the project that are to be satisfied.

Here is a link to my [project code](https://github.com/mymachinelearnings/CarND-Path-Planning-Project/)

---
### What is Semantic Segmentation
Semantic segmentation is understanding an image at pixel level. Simply put, it is the process of understanding what is in an image at pixel level. 
Each pixel of the image will be assigned a corresponding class. This is really useful in preception systems. 
With respect to autonomous driving systems, there are various applications for Semantic Segmentation like the need to understand the scene around 
the car and perceive what's around it - whether its another car, or a pedestrian, a traffic light, or a static obstacle, or more importantly, the drivable portion of the road

### Architecture

Semantic Segmenation can be achieved by Fully Convolutional Networks(FCN)

In the normal convolutional neural networks(CNN) used for classification, the architecture would be a series of convolutional/non-linear activation/pooling layers 
followed by Fully Connected(Dense) layers to classify the objects in the image. The Fully Connected layers extract the 'what' portion while losing the details about the
 'where' portion(spatial information). Also due to the fact that fully connected layers are of a specific size, the inputs to these kinds of classifiers are usually fixed. 
It cannot operate on different sizes of inputs. 
 
If we replace the fully connected layers with a series of transpose convolutional layers(meaning upsampling from samller to larger size while doing reverse convolution), then the output would be of the same size as the input, and since there are no dense/fully connected layers, the spatial information is saved and we can estimate both the 'what' and 'where' portions of the image. This way, each pixel could be assigned a class.

There are 3 main parts to FCN's

- Encoder part followed by a 1x1 convolution
- Decoder part
- Skip Connections

The encoder consists of a series of Convolutional layers (with non-linear activations & Pooling layers) that gradually minimizes the state space. 
This module is responsible for determining the 'what' portion of hte object - meaning to classify the object's class in the image

The decoder consists of a series of transpose convolutional layers (upsampling layers) that retain the size of the image to its original shape while determining the 'where' portion of the image - the spatial information that's lost during encoder will be retained

Skip connections enhance the spatial information so that the final segmentation is more accurate. Skips connections are implemented by element-wise addition of a decoder layer
with its corresponding encoder layer. Note that the shpaes of these two should be same for matrix addition to happen

### Implementation
I've used Tensorflow 1.x with Python 3.6 to implement the solution.

The encoder part is taken from a pretrained model on VGG6 trained on ImageNet. Encoder had 5 pooling layers of 2x2 Kernel which means the image must have been squeezed by 32x in its height x width
The final dense layers of VGG are replaced by a 1x1 convolution layer. 
This is followed by a transpose convolution (upsampling) by the same factor as it was downsampled in the encoder portion(32x). This is done in 3 stages with Kernel Sizes of 4, 4, 16 and Strides of 2, 2, 8 respectively.

To enhance the segmentation, the 4th layer of VGG Encoder is connected to the first decoder layer, 5th layer of VGG Encoder is connected to the second decoder layer. When I say connected, it means it is the addition of the skip layer with the its previous layer

The final activation loss is calculated as the cross entropy softmax and the network is trained with a learning rate of 0.001 for different epochs to compare the results

It is clearly observed that the accuracy (measured here by th IOU metric) is better with increasing number of epochs. I tried for 2, 5, 20 epochs and here are the loss values after each epoch

|     Epochs    |     Loss      |
| ------------- | ------------- |
| 2 Epochs      | 0.186         |
| 5 Epochs      | 0.119         |
| 20 Epochs     | 0.025         |




### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
