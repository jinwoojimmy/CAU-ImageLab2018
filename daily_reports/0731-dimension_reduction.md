# 31/07/18 Daily Report


## Dimensionality Reduction

### Basic Background
  * Dimensionality :
  
    Dimensionality refers to the minimum number of coordinates needed to specify any point within a space or an object. So a line has a dimensionality of 1 because only one coordinate is needed to specify a point on it.

### Definition
Dimensionality reduction is a series of techniques in machine learning and statistics to reduce the number of random variables to consider. 
It involves feature selection and feature extraction. 
Dimensionality reduction makes analyzing data much easier and faster for machine learning algorithms without extraneous variables to process, 
making machine learning algorithms faster and simpler in turn.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/dim_reduction.jpg" width="500" height="200">

### Curse of Dimensionality
The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces (often with hundreds or thousands of dimensions) that do not occur in low-dimensional settings such as the three-dimensional physical space of everyday experience.
The expression was coined by Richard E. Bellman when considering problems in dynamic optimization.
  
* Problem of high dimension
  - High computational cost
  
  - Overfitting problem ; just suitable for training data, which means difficult to figure out good feature.
  
  - Difficulty in visualization

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/dim_performance.PNG" width="400" height="400">
The important information can be extracted with lower information. So we can try dimensional reduction!

### Utilization
  * Data mining
  * Knowledge discovery 
  
  etc

### Manifold & Manifold Learning

#### Manifold
The subspace which well-represents samples of data in space.

#### Manifold Learning
The data set lies along a low-dimensional manifold embedded in a high-dimensional space,
where the low-dimensional space reflects the underlying parameters and high-dimensional space is the feature space.
Attempting to uncover this manifold structure in a data set is referred to as manifold learning.

Simply, this is the process of figuring out appropriate euclidean space from the collected data.

By doing so, we can express the given information into more compact, meaningful and effective way.

And, manifold learning is a non-linear dimensionality reduction technique.

The most famous way of linear dimensionality reduction is **PCA(Principal Component Analysis)**.

