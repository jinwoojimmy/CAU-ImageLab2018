# Research Report 
    
## About

The report contains studies, discussion and notes based on research performed in Image Lab, Chung-Ang University at 2018 summer.

## Goals

The purpose of our program is to
* Understand basic knowledge of machine learning and deep neural network 
* Comprehend state-of-the-art techniques and models in deep learning
* Design and implement denoising model with good performance 

## Research Plan


### 1st Week, September
* Graduate Paper
* Seminar on recent research paper 
* Study on paper


### 2nd Week, September
* Seminar on recent research paper 
* Study on paper


### 3rd Week, September
* Seminar on recent research paper 
* Study on paper


### 4th Week, September
* Seminar on recent research paper 
* Study on paper


### 1st Week, October
* Seminar on recent research paper 
* Study on paper


### 2nd Week, October
* Seminar on recent research paper 
* Study on paper


### 3rd Week, October
* Seminar on recent research paper 
* Study on paper


### 4th Week, October
* Seminar on recent research paper 
* Study on paper



# Graudate Paper

This section is about paper related to graduation.

The paper's topic is *Progress of Configuration in Android
on Pre-trained Deep Learning Model
Implemented by TensorFlow*

## 1 Introduction

As Deep Learning technology has developed
rapidly recently, it has shown positive performance
in various fields. The deep learningbased
system has shown excellent performance
in areas that were considered impossible in
the past, such as AlphaGo, who won the Go
match, and Google duplex, which understand
the context of dialogue and engage in conversation
naturally. Furthermore, efforts are being
made to mount Deep Learning models on
smartphones used by many people such as
[MobileNets](https://arxiv.org/abs/1704.04861). Various deep learning libraries
are being released to implement deep learning,
and [TensorFlow](https://arxiv.org/abs/1603.04467), developed by Google,
provides an interface for deep learning in mobile
devices. Smart phones using the Android
platform went further on Android Studio, and
Google launched the TensorFlow Lite, a lighter
solution. In this paper, the goal is to explore
how to use the previously learned deep learning
model in Android.



## 2 BACKGROUND

### 2.1 TensorFlow
TensorFlow is an open-source library created by
Google for machine learning and deep learning
that is configured to run quickly. A flexible
architecture can be adopted and deployed on
one or more CPUs or GPUs, and can be applied
to a wide range of areas as well as machine
learning and neural network research.


### 2.2 TensorFlow Lite
TensorFlow Lite is a solution for making TensorFlow
lightweight for mobile and embedded
devices. Supports hardware acceleration with
existing TensorFlow and end points, and can
convert implemented TensorFlow models to
perform at low capacity and at high speed.


### 2.3 TensorFlowInferenceInterface
Java interface class that Google implements to
call up and use a model implemented with
TensorFlow using Java. In September 2016, a
commit for that class was raised for the first
time. It provides a convenient function to use
TensorFlow on an Android platform.


### 2.4 Protocol Buffers File
A file with an extension of ’.pb’ is a protocol
buffers-type file that is a serialized data structure
developed by Google and released on open
source. TensorFlow can extract the graph form
of the learned model in the form of ’.pb’ and the
values of parameters obtained after training can
be extracted into a ’.ckpt’ file. These two files
can be combined as new ’.pb’ files containing
the graph and parameter values of the learned
deep learning model at the same time. In this
paper, we will assume that ’.pb’ file as the
newly freezed graph from now on.


### 2.5 Android NDK and JNI
Android NDK is a set of tools that allows
Android to utilize native code such as C and
C++. They are often used when a library that
is implemented in other languages is reused
or operations with heavy calculations, such as
physics simulations, are required to be used at
higher performance. JNI provides an interface
for calling other languages within Java.


### 2.6 TensorFlow in Android Studio
Android Studio is an integrated development
tool for developing Android platform-based
applications, enabling you to install the libraries
and dependencies that applications
need. Libraries for running TensorFlow can also
be easily set up and deployed in Android Studio.


### 2.7 ML Kit
ML Kit is an iOS and Android SDK that was
released by Google that provides variety of
Machine Learning APIs through Firebase. It
is possible to infer using the TensorFlow Lite
and experiment with other machine learning
models using A/B testing through the interface
with the Firebase. It is currently registered as a
beta version.


## 3 MAIN
### 3.1 Native Code Implementation
Before Google unveiled the TensorFlowInreferenceInterface,
developers had to write native
code to run on Android platforms based on
some of the TensorFlow codes written in C++.
Because it was necessary to implement code
that operated by reading ’.pb’ file containing
information from learned deep learning, Android
developers had difficulty on accessing
and implementation.


### 3.2 Advent of TensorFlowInferenceInterface
As Google introduced the TensorFlow-
InreferenceInterface, developers can
handle deep learning only with Java
code without writing native code.
’libandroid tensorflow inference java.jar’,
providing a Java interface and ’libtensorflow
inference.so’, a native library of JNI
codes for TensorFlow provides easy way to
mount models and make inferences about
given inputs. By May 2017, all necessary APIs
could be installed with simple inputs from the
Gradle, a project distribution tool.


### 3.3 Optimization using TensorFlow Lite
The advent of the TensorFlow Lite has allowed
the company to reduce the size and speed
of the Deep Learning network. Existing ’.pb’
files could be converted to ’.tflite’ files using
the TensorFlow Lite TOCO script and mounted
through the Interpreter class provided by the
TensorFlow Lite.


### 3.4 ML Kit
ML Kit is a tool that provides a number of
APIs based on the TensorFlow Lite model, and
uploads the model for use in post-host applications.
The FireBaseLocalModelSource class
allows you to call in models that exist in your
local area, or you can also use the FireBase-
CloudModelSource to register in remote areas.


## 4 CONCLUSION
As Deep Learning has developed in many areas,
various methods have been employed for
using Deep Learning models in smart phones.
Google, which is responsible for the Android
platform, has constantly updated how developers
can use the easy-to-use libraries to implement
deep learning-based applications. In this
paper we looked at the transition process of
using deep learning based on Android. In the
future, more developers will be able to lower
the barriers to deep learning and create more
practical deep learning-based applications.


## REFERENCES
[1] A. Howard, M. Zhu et al. MobileNets: Efficient Convolutional
Neural Networks for Mobile Vision Applications
arXiv:1704.04861, 2017


[2] M. Abadi, A. Agarwal et al. Tensorflow:Large-scale
machine learning on heterogeneous distributed systems
arXiv:1603.04467, 2016


## Reference Image 

This part wass not included in paper. In order to help easy understanding, this section was inserted.
### Tensorflow Lite Architecture
![image](https://github.com/jwcse/DeepLearning/blob/master/img/tensorflowlite.png)

### JDK & JNI
![image](https://github.com/jwcse/DeepLearning/blob/master/img/JNI_NDK.png)





