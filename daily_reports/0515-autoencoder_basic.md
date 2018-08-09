# 15/05/18 Daily Report

## Basic Study on Autoencoder
Machine learning can be divided into two part. One is supervised learning and the other one is unsupervised learning. 
In case of supervised learning, input data for training includes class label. The class label represents which group does the given
instance of data belongs. Therefore, it needs effort for labeling on each data. On the other hand, unsupervised learning does not
need such effort, and feature is extracted by relation between dataset. 

Autoencoder is one of the most famous example of unsupervised learning, and this is what we are going to look at.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/AE_overview.PNG" width="700" height="350">


  Let's consider input data (![equation](https://latex.codecogs.com/gif.latex?x%5E%7B%28i%29%7D), ![equation](https://latex.codecogs.com/gif.latex?y%5E%7B%28i%29%7D)) ,
which is i-th instance's attribute among input data.
In specific, ![equation](https://latex.codecogs.com/gif.latex?x%5E%7B%28i%29%7D) refers i-th instance's attributes, and ![equation](https://latex.codecogs.com/gif.latex?y%5E%7B%28i%29%7D) refers i-th instance's class among input data.

We can define hypothesis function which takes ![equation](https://latex.codecogs.com/gif.latex?x%5E%7B%28i%29%7D) as input and ![equation](https://latex.codecogs.com/gif.latex?y%5E%7B%28i%29%7D) as output.  
The size of ouput result is equal to input size. In progress of encoding, input is compressed passing hidden layer. With this compressed property, 
autoencoder is utilized as Feature extracter.

## Training of Autoencoder
1) With calculated input and hidden layers' weights, pass through sigmoid function.
2) Calculating output of 1) and output layer's weights, pass through sigmoid function. 
3) With 2), calculate MSE(Mean Squared Error).
4) With 3), optimise loss with SGD(Stochastic Gradient Descent).
5) Update weights by backpropagation.

## Sigmoid function

 ![equation](https://latex.codecogs.com/gif.latex?y%3D%5Cfrac%7B1%7D%7B1&plus;e%5E%7B%28ax&plus;b%29%7D%7D)
* If ![equation](https://latex.codecogs.com/gif.latex?a) decreases, error increases and if ![equation](https://latex.codecogs.com/gif.latex?a) increases, error decreases.
* If ![equation](https://latex.codecogs.com/gif.latex?b) is too big or small, error increases.
* ![equation](https://latex.codecogs.com/gif.latex?y) value is betwwen 0 and 1. So, if real value is 1 but predict value is close to 0, error should increase. Likewise, if real value is 0 but predict value is close to 1, error should increase.
## SGD(Stochastic Gradient Descent)


## Backpropagation


## Derivation of Autoencoder
* Stacked Auto Encoder : Composed of multiple layers

* Sparse Auto Encoder : Hidden layer part is bigger than input an output part.

* Denoising Auto Encoder : For given noised data, denoise and represent original data.



