# 16/05/18 Daily Report



## Gradient Descent
"Finding ![equation](https://latex.codecogs.com/gif.latex?w) that minimizes the loss"

![equation](https://latex.codecogs.com/gif.latex?loss%28w%29%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28%5Chat%7By_%7Bn%7D%7D%20-y_%7Bn%7D%29)

To specify, it is way of finding the point which minizes the loss function, whose derivative value at the point is equal to zero.

The process of finding the point ![equation](https://latex.codecogs.com/gif.latex?m) is like this :

On a graph, calculate gradient(derivative) value at certain point ![equation](https://latex.codecogs.com/gif.latex?a_%7B1%7D).
Then, using equation

![equation](https://latex.codecogs.com/gif.latex?w%20%3D%20w%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20loss%7D%7B%5Cpartial%20w%7D),
move ![equation](https://latex.codecogs.com/gif.latex?m) until we reach to the point ![equation](https://latex.codecogs.com/gif.latex?m).  

In above equation, gradient is equal to ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20loss%7D%7B%5Cpartial%20w%7D).

The picture below is the screenshot for easy understanding.
  <img src="https://github.com/jwcse/DeepLearning/blob/master/img/GradientDescent1.png" width="600" height="400">

#### Implementation of Gradient Descent in Python
[Gradient Descent code](./codes/gradientDescent.py)




## Back propagation
In case of simple perceptron(one input and one output), we could optimize the coefficient with gradient descent.
; after calculating by original model, we could update the original model's coefficient by gradient descent.

But what if there are a lot of layers between input layer and output layer?

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/backpropagationWHY.png" width="500" height="400">

In this case we can use **chain rule**.

The **chain rule** is equal to combining each node's derivative in order to get the derivative of far-away node's value.  

1. With initial weights, calculate the output(forward).
2. With output, calculate the difference between expected value.
3. With gradient descent, update the former(close to input) weight.
4. iterate 1-3 until loss cannot be shrinked.


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/backpropagation2.png" width="500" height="400">



