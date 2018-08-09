# 23/07/18 Daily Report

## Seminar

Undergraduate students and several graduate students in Image Lab gathered to hold a seminar.

Each undergraduate student prepared own section with selected paper chosen by individual preferences.

In addition, each student complemented last week's presentation by studying more on mis-understood part on last week. 

### Saliency-Guided Unsupervised Feature Learning for Scene Classification

Jinwoo(Jeon) presented on paper [Saliency-Guided Unsupervised Feature Learning for Scene Classification](https://ieeexplore.ieee.org/document/6910306/)


#### Intro
In the paper, researchers propose an unsupervised feature learning framework for scene classification.

By using the saliency detection algorithm, researchers extract a representative set of patches from the salient regions in the image data set. 

These unlabeled data patches are exploited by an unsupervised feature learning method to learn a set of feature extractors which are robust and efficient and do not need elaborately designed descriptors such as the scale-invariant-feature-transform-based algorithm. 

#### Traditional Scene Classification Methods
  * K-means clustering
  
  * Bag of Visual Words (BoW)
  
    The clustering results of local features extracted from image is placed in codebook.
    
    Find out the nearest code-word with the feature extracted from image, and object is classified based on representation of historgram.
     
  * Spatial Pyramid Matching Kernel(SPMK)
  
    This is alternative for BoW that BoW loses spatial information when presenting into histgram.
    
    After dividing region and extracting histogram, the result is asssembled again and pyramid is built. Then, classification is proceeded 
    comparing with the pyramid with another one.  
  
  * Sparse Coding


#### Saliency-Guided Sampling

#### Unsupervised Feature Learning
  * Sparse Autoencoder
  
  * Kullback-Leibler Divergene

#### Scene Classificatino via SVM






### ANN(Artificial Neural Network)

Ki-soo presented on ANN.

He introduced concept of perceptron, necessity of MLP and a role of the bias.



### BM3D Filter

Yeon-ji presented on BM3D filter.


### ML on Android

Jinwoo(Kim) presented on ML on Android by showing result of ML kit project.


