# Udacity Self-Driving Car Engineer Nanodegree

## Term 3 Project 2 : Semantic Segmentation
### Project Writeup

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---
This document explains the Semantic Segmentation project that involves building a Fully Convolutional Network(FCN) and discusses the architecture. The outcome of this project will be a pixel wise labelling of the input image

---
### Rubric Points
Here are the [rubric points](https://review.udacity.com/#!/rubrics/989/view) for the project that are to be satisfied.

Here is a link to my [project code](https://github.com/mymachinelearnings/CarND-Semantic-Segmentation)

---
### What is Semantic Segmentation

![Semantic Segmentation Image](data/WriteupImages/20epoch_3.png)


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

### Results

#### Skip Connections

It is clear that Skip connections have increased the accuracy of the segmentation, here I'm comparing the same images with and without Skip Connections

|     With Skip Connection                                              |     Without Skip Connections                                                |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| ![Semantic Segmentation Image](data/WriteupImages/WithSkip1.png)      | ![Semantic Segmentation Image](data/WriteupImages/WithoutSkip1.png)         |
| ![Semantic Segmentation Image](data/WriteupImages/WithSkip2.png)      | ![Semantic Segmentation Image](data/WriteupImages/WithoutSkip2.png)         |

#### Training with different Epochs

The trainable parameters of a network will be adjusted during each epoch in a training. There's threshold to the number of epochs to be used, but the general idea is that the 
accuracy increases with the number of epochs during training (up to a certain level after which it becomes more or less constant)

Here, I've tried with 2, 5, 20 Epochs and compared the results and its clear the higher epochs gave better results

|     2 Epochs                                                          |     5 Epochs                                                                |     20 Epochs                                                               |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| ![Semantic Segmentation Image](data/WriteupImages/2epoch_1.png)      | ![Semantic Segmentation Image](data/WriteupImages/5epoch_1.png)              | ![Semantic Segmentation Image](data/WriteupImages/20epoch_1.png)            |
| ![Semantic Segmentation Image](data/WriteupImages/2epoch_2.png)      | ![Semantic Segmentation Image](data/WriteupImages/5epoch_2.png)              | ![Semantic Segmentation Image](data/WriteupImages/20epoch_2.png)            |
| ![Semantic Segmentation Image](data/WriteupImages/2epoch_3.png)      | ![Semantic Segmentation Image](data/WriteupImages/5epoch_3.png)              | ![Semantic Segmentation Image](data/WriteupImages/20epoch_3.png)            |


---

### Installation & Setup
---

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

---

### **Authors** <br/>
* Ravi Kiran Savirigana

### **Acknowledgements** <br/>
Thanks to Udacity for providing the startup code to start with. And a great community help from stackoverflow.com & github.com
