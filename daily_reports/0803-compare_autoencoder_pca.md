# 03/08/18 Daily Report

## An autoencoder with linear transfer functions is equivalent to PCA

Let’s prove the equivalence for the case of an autoencoder with just 1 hidden layer, the bottleneck layer.

First recall how pca works:

![equation](https://latex.codecogs.com/gif.latex?x) the original data, ![equation](https://latex.codecogs.com/gif.latex?z) the reduced data and ![equation](https://latex.codecogs.com/gif.latex?z) the reconstructed data from the reduced representation. 

Then we can write pca as:

![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20z%3DB%5E%7BT%7Dx%20%5Cnewline%20%5Chat%7Bx%7D%20%3D%20Bz%20%5Cnewline)




Now consider an autoenoder:

Consider the architecure in the picture
<img src="https://github.com/jwcse/DeepLearning/blob/master/img/cmp_autoencoder_pca.PNG" width="200" height="300">

Which basically is ![equation](https://latex.codecogs.com/gif.latex?x%5Crightarrow%20z%5Crightarrow%20%5Chat%7Bx%7D)

Since we said that the activation functions are linear transfer functions ![equation](https://latex.codecogs.com/gif.latex?%5Csigma%28x%29%3Dx), then we can write the autoencoder as:

![equation](https://latex.codecogs.com/gif.latex?%5Chat%7Bx%7D%3DW_%7B1%7DW_%7B2%7Dx)

where ![equation](https://latex.codecogs.com/gif.latex?W_%7B1%7D) and ![equation](https://latex.codecogs.com/gif.latex?W_%7B2%7D) are the weights of the first and second layer.

Now if we set ![equation](https://latex.codecogs.com/gif.latex?W_%7B1%7D%20%3D%20B%2C%20W_%7B2%7D%20%3D%20B%5E%7BT%7D) we have:

![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20%5Chat%7Bx%7D%3DW_%7B1%7D%28W_%7B2%7Dx%29%5Cnewline%20%5Chat%7Bx%7D%3DW_%7B1%7Dz%20%5Cnewline%20%5Chat%7Bx%7D%3DBz)

which is the same solution that we had for PCA.

Note that the equivalence is valid only for autoencoders that have a bottleneck layer smaller than the input layer.
