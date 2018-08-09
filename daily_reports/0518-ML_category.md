# 18/05/18 Daily Report



## Machine Learning

  * *"Field of study that gives computers the ability to learn without being explicitly programmed" - Arthur Samuel*
  
  * Machine learning is a subset of artificial intelligence in the field of computer science that often uses statistical techniques to give computers the ability to "learn" (i.e., progressively improve performance on a specific task) with data, without being explicitly programmed 
 [from Wikipedia](https://en.wikipedia.org/wiki/Machine_learning)
 
### Category

There are mainly three categories in machine learning.

#### Supervised Learning

##### Overview
According to dictionary, supervisor has meaning of instructor who leads and teaches a student.

In this context, we can say that the supervised learning means to predict based on past knowledge. 

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/supervised_learning_overview.PNG" width="800" height="400">


In order to obtain good training result, there should be a lot of training data, and the data should have generalization property.

##### Procedure

1. Select the training data for training


2. Collect the training data

3. Decide how to represent the feature of the input

  * Usually represented in vector form.

  * Curse of dimensionality should be considered ; number of feature shouldn't be too large.

4. Decide training algorithm 

  * There are a lot of algorithms and their properties are also diverse. Appropriate algorithm should be selected based on purpose.  

5. Train with the data and algorithm

6. Evaluate the accuracy of the designed model

##### Supervised Learning Algorithms

These are representative algorithms of supervised learning.

  - Artifical neural network
  
  - Boosting
  
  - Bayesian statistics
  
  - Decision tree
  
  - Gaussian process regression
  
  - Nearest neighbor algorithm
  
  - Support vector machine
  
  - Random forests
  
  - Symbolic machine learning
  
  - Ensembles of classifiers
  


#### Unsupervised Learning

In case of supervised learning, computing error function or loss function is possible through training.

In other words, model can be updated with feedback. However, for unsupervised learning, this is not possible.


##### Overview

Unsupervised machine learning is the machine learning task of inferring a function that describes the structure of "unlabeled" data 
(i.e. data that has not been classified or categorized).

Since the examples given to the learning algorithm are unlabeled, there is no straightforward way to evaluate the accuracy of the structure that is produced by the algorithm—one feature that distinguishes unsupervised learning from supervised learning and reinforcement learning.

[Wiki](https://en.wikipedia.org/wiki/Unsupervised_learning)


##### Category

  * Clustering
    - k-means
    - mixture models
  
  * Neural netowrk
    - Autoencoder
    - Hebbian Learning
    
  * Dimensionality reduction
    - PCA
    - ICA
    - SVD
    - Non-negative matrix factorization


The picture below shows an example of unsupervised learning ; *k-means-clustering*.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/k-means-clustering.png" width="700" height="400">




#### Reinforcement Learning

Reinforcement learning (RL) is an area of machine learning, inspired by behaviorist psychology[citation needed],
concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. 

Like training a dog with reward and punishment, training is performed so as to get reward as much as possible. 

This is being researched on game theory, control theory, simulation-based optimizatino, multi-agent systems, genetic systems and so on.

