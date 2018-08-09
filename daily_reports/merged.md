# Research Report 
    
## About

The report contains studies, discussion and notes based on research performed in Image Lab, Chung-Ang University at 2018 summer.

## Goals

The purpose of our program is to
* Understand basic knowledge of machine learning and deep neural network 
* Comprehend state-of-the-art techniques and models in deep learning
* Design and implement denoising model with good performance 

## Research Plan

### 3rd Week, May
* 1st meeting with mentors and professor for research plan
* Environment configuration and setting
* Study basic knowledge for deep learning

### 4th Week, May
* Study basic knowledge for deep learning

### 5th Week, May (1st Week, June)
* Meeting with mentors and professor
* Research on CNN
     
### 2-4 Week, June
* Study depending on personal schedule ; mid-term period 

### 5th Week, June
* Meeting with mentors and professor to figure out research goal 
* Seminar on recent research paper 
* Study on paper

### 1st Week, July
* Seminar on recent research paper 
* Study on paper

### 2nd Week, July
* Seminar on recent research paper 
* Study on paper
* Implement deep learning on android

### 3rd Week, July
* Seminar on recent research paper 
* Study on paper

### 4th Week, July
* Seminar on recent research paper 
* Study on paper

### 5th Week, July
* Seminar on recent research paper 
* Study on paper

### 1st Week, August
* Seminar on recent research paper 
* Study on paper

### 2nd Week, August
* Seminar on recent research paper 
* Study on paper

### 3rd Week, August
* Seminar on recent research paper 
* Study on paper
* Design web application for labeling

### 4th Week, August
* Seminar on recent research paper 
* Study on paper
* Design web application for labeling

### 5th Week, August
* Implement web application for labeling

# 0514-meeting_setting.md

# 14/05/18 Daily Report

## Meeting
- Research Topic Decision
  Under project managing graduate student's direction, had a meeting to figure out which topic to focus on with other students.
We decided to conduct research on 'Autoencoder'. We divided the topic into three parts, and I was delegated to research on 
'data augmentation' part. Autoencoder is like ordinary neural network's unsupervised learning version. After two weeks' study and 
research on Autoencoder, we aim to write a paper about it.
- Technologoy Stack
  We decided to use Python as main language and use pytorch library under Anaconda3 distribution. Students will be provided with
user account to access linux server which contains GPU and appropriate for running machine learning program.
- Study Guideline  
  After deciding the research topic, each students received links for study and research from the graduate student. 
  
## Environment Configuration and Installation
- install anaconda3
- install pytorch
- install PyCharm IDE

## Study on Pytorch Basics

  - Pytorch includes various kinds of computational library and methods.

```python
  
import torch
import numpy as np
import torch.nn.init as init

#
#   1. CREATE TENSOR
#
#  1) random numbers
# random numbers
x = torch.rand(2, 3)

# random signed numbers
# mean = 0, variance = 1 standard normal distribution following random number
x = torch.randn(2, 3)

# permutation of 0 - n
x = torch.randperm(5)


# 2) zeros, ones, arange
# array with r, c size with all element 1
x = torch.ones(2, 3)

# [start, end]with given step
x = torch.arange(0, 3, step=0.5)

# 3) Tensor Data Type
# python list type to FloatTensor type
x = torch.FloatTensor([2, 3])

# tensor.type_as(tensor_type)
x = x.type_as(torch.IntTensor())

# 4) Numpy to Tensor, Tensor to Numpy
x1 = np.ndarray(shape=(2, 3), dtype=int, buffer=np.array([1, 2, 3, 4, 5, 6]))
x2 = torch.from_numpy(x1)

x3 = x2.numpy()

# 5) Tensor on CPU & GPU
x = torch.FloatTensor([[1,2,3],[4,5,6]])
# gpu
# x_gpu = x.cuda()      # torch.cuda.FloatTensor should be enabled
# cpu
# x_cpu = x_gpu.cpu()

# 6) Tensor Size
x = torch.FloatTensor(10, 12, 3, 3)

# print(x.size())     # [10, 12, 3, 3]
# print(x.size()[1:])     # [12, 3, 3] ; indexing is possible


#
#   2. Indexing, Slicing, Joining, Reshaping
#
# 1) Indexing
x = torch.rand(4, 3)
# torch.index_select
out = torch.index_select(x, 0, torch.LongTensor([0, 3]))
# print(x, out)

# pythonic indexing
x[:, 0], x[0,:], x[0:2, 0:2]

# torch.masked_select
x = torch.randn(2, 3)
mask = torch.ByteTensor([[0,0,1],[0,1,0]])
out = torch.masked_select(x,mask)

# x, mask, out

# 2) Joining
# torch.cat(seq, dim=0)     concatenate tensor along dim
# 1 2 3
# 4 5 6
x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# -1 -2 -3
# -4 -5 -6
y = torch.FloatTensor([[-1, -2, -3], [-4, -5, -6]])

# 1 2 3
# 4 5 6
# -1 -2 -3
# -4 -5 -6
z1 = torch.cat([x, y], dim=0)

# 1 2 3 -1 -2 -3
# 4 5 6 -4 -5 -6
z2 = torch.cat([x, y], dim=1)

# torch.stack(sequence, dim=0)      stack along new dim
x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])

x_stack = torch.stack([x, x, x, x], dim=0)

# 3) Slicing
# torch.chunk(tensor, chunks, dim=0)    tensor into num chunks
x_1, x_2 = torch.chunk(z1, 2, dim=0)
y_1, y_2, y_3 = torch.chunk(z1, 3, dim=1)

# torch.split(tensor, split_size, dim=0)    split into specific size
x1, x2 = torch.split(z1, 2, dim=0)
y1, y2 = torch.split(z1, 2, dim=1)  # y1 contains 2-dim-vector and y2 contains 1-dim-vector

# 4) Squeezing  - reduce dim by 1
x1 = torch.FloatTensor(10, 1, 3, 1, 4)
x2 = torch.squeeze(x1)      # [10, 3, 4]

# unsqueezing   - add dim by 1
x1 = torch.FloatTensor(10, 3, 4)
x2 = torch.unsqueeze(x1, dim=0)     # size = [1, 10, 3, 4]

# 5) Reshaping
# rensor.view(size)
x1 = torch.FloatTensor(10, 3, 4)    # .size = [10, 3, 4]
x2 = x1.view(-1)                    # .size = 120
x3 = x1.view(5, -1)                 # .size = [5, 24]
x4 = x1.view(3, 10, -1)             # .size = [3, 10, 4]

#
#   3. Initialization
#
x1 = init.uniform_(torch.FloatTensor(3, 4), a=0, b=9)
x2 = init.normal_(torch.FloatTensor(3, 4), std=0.2)
x3 = init.constant_(torch.FloatTensor(3,4),3.1415)

#
#   4. Math Operations
#
# 1) Arithmetic Operations
torch.add(x1, x2)   # two tensors' size should be equal
x3 = x1 + 10     # plus ten on each element of x1

torch.mul(x1, x2)   # multiply element to element
x3 = x1 * 10    # multiply ten on each element of x1

torch.div(x1, x2)   # divide element to element
x3 = x1 / 5

# 2) Other Math Operations
torch.pow(x1, 2), x1**2

torch.exp(x1)

torch.log(x1)   # natural logarithm

# 3) Matrix Operations
torch.mm(torch.FloatTensor(3, 4), torch.FloatTensor(4, 5))  # matrix multiplication

torch.bmm(torch.FloatTensor(10, 3, 4), torch.FloatTensor(10, 4, 5))   # batch matrix multiplication

torch.dot(torch.FloatTensor([1, 4]), torch.FloatTensor([1, 4]))   # dot product of two tensor

xT = x1.t()         # transpose


torch.eig(torch.FloatTensor(4, 4), True)   # eigen value, eigen vector
```

# 0515-autoencoder_basic.md
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




# 0516-GD_backpro.md
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

### Implementation of Gradient Descent in Python
```python

"""
    This is for understanding the concept "Gradient Descent"
    We assume that 
        hypothesis function :
            y = w * x
    
    We have to find out 'w' which minimizes the loss,  
    so that we can get appropriate function
    
    In real training, process of gradient descent is executed until convergence.
    But in this practice, we iterate just until enough to watch almost converged.
    
"""

# just random value
w = 3.0
# learning rate - just randomly selected
LR = 0.01

x_data = [1.0, 2.0, 3.0]    # x1, x2, x3
y_data = [2.0, 4.0, 6.0]    # y1, y2, y3


# our model's forward pass
def forward(x):
    return x * w


# loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)      # (w * x - y)^2


# compute gradient on loss function
def gradient(x, y):
    return 2 * (w * x - y) * w


# run program
def run():
    # use global variable w and update in the function
    global w

    print("predict (before training)", 4, forward(4))

    # Training loop ; train 15 times
    for epoch in range(15):
        # train once with given set of data
        for x_val, y_val in zip(x_data, y_data):    # (xi, yi)
            grad = gradient(x_val, y_val)
            w = w - LR * grad
            print("\tgrad: ", x_val, y_val, round(grad, 2))
            l = loss(x_val, y_val)

        print("progress{num}:".format(num=epoch), " w=", round(w, 2), "loss=", round(l, 2))

    # After training
    print("predict (after training)",  "4 hours", forward(4))


if __name__ == "__main__":
    run()


```




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




# 0518-ML_category.md
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


# 0521-autogradient_regression.md
# 21/05/18 Daily Report

## Auto Gradient

*torch.Tensor* is the central class of the package. If we set its attribute .requires_grad as True, it starts to track all operations on it. When we finish our computation we can call .backward() and have all the gradients computed automatically. The gradient for this tensor will be accumulated into .grad attribute.

If you want to compute the derivatives, you can call .backward() on a Tensor as mentioned above. If Tensor is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments to backward(), however if it has more elements, you need to specify a gradient argument that is a tensor of matching shape.


We've seen gradient descent operating by our handling. The code below is implementation of gradient descent using pytorch library.

### Code 1

This code is from [hunkim's code](https://github.com/hunkim/PyTorchZeroToAll/blob/master/03_auto_gradient.py).

```python

import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]),  requires_grad=True)  # Any random value

# our model forward pass


def forward(x):
    return x * w

# Loss function


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# Before training
print("predict (before training)",  4, forward(4).data[0])

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print("progress:", epoch, l.data[0])

# After training
print("predict (after training)",  4, forward(4).data[0])



```


### Code 2

```python
import torch
from torch.autograd import Variable


# 1) Declaration

x_tensor =torch.Tensor(3, 4)    # FloatTensor of size 3x4

x_variable = Variable(x_tensor)

# 2) Variables of a Variable

# .data -> wrapped tensor
x_variable.data

# .grad -> gradient of the variable
x_variable.grad

# .requires_grad -> whether variable requires gradient
x_variable.requires_grad

# .volatile -> inference mode with minimal memory usage
x_variable.volatile


# 3) Graph & Variables
x = Variable(torch.FloatTensor(3, 4), requires_grad=True)
y = x**2 + 4*x  # requires_grad = True
z = 2*y + 3     # requires_grad = True

# .backward(gradient, retain_graph, create_graph, retain_variables)
# compute gradient of current variable w.r.t. graph leaves

gradient = torch.FloatTensor(3, 4)
z.backward(gradient)
# x.grad, y.grad, z.grad


```

## Linear Regression in Pytorch

By using pytorch, process of forward, backwardpropagation and updating for optimization becomes easier.
Let's look at the *Linear Regression* as example. The process of coding in pytorch can be divided into three part :

### 1. Design model using class with Variables
Neural networks can be constructed using *torch.nn* package.
*nn* depends on 'autograd' to define models and differentiate them.

An *nn.Module* contains layers, and a method *forward(input)* that returns 'output'.

A typical training proceduer for a neural network is as follows :
- Define the neural network that has some learnable parameters(or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss(how far is the output from being correct)
- Propagate gradients back into the network's parameters
- Update the weights of the network, typically using a simple update rule :
'weight = weight - learning_rate * gradient'

### 2. Construct loss and optimizer

### 3. Training cycle (forward, backward, update)


```python
import pytorch
from torch.autograd import Variable

# Data definition
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


# Model class in Pytorch way
class Model(torch.nn.Module):
  def __init__(self):
  """
    In the constructor we instantiate two nn.Linear module
  """
  super(Model, self).__init__()
  self.linear = torch.nn.Linear(1, 1) # One in and one out
  
  def forward(self, x):
    """
      In the forward function we accept a Variable of input data and we must return a Variable of output data.
      We can use Modules defined in the constructor as well as arbitrary operators on Variables.   
    """
      y_pred = self.linear(x)
      return y_pred

# our model
model = Model()
  
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # parameter : (what need to be updated, learning_rate)

# Training loop
for epoch in range(500):
        # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After training
hour_var = Variable(torch.Tensor([[4.0]]))
y_pred = model(hour_var)
print("predict (after training)",  4, model(hour_var).data[0][0])
```





# 0522-sigmoid.md
# 22/05/18 Daily Report

## Logistic Regression in Pytorch
"Prediction of zero or one (pass or fail)"

Example : 
  - Spent N hours for study -> pass or fail?
  - GPA and GRE scores for CMU PHD program -> admit or not?
  - She/he looks good -> propose or not?
  - Soccer game against Japan -> win or not?

In order to predict binary decision, **sigmoid** function works great.
  
### Sigmoid
Sigmoid function is one of representatic activation function.
In neural net, activation function is equal to function **F** in equation **y=F(x)** which transmits output y to next layer, for x=Weighted Sum.

Basic idea of sigmoid function is to squash number between zero to one.
The function equation is like this : [link](https://github.com/jwcse/DeepLearning/blob/master/daily_reports/150518.md#sigmoid-function)

In order to apply, we can just wrap our model function with sigmoid function.
However, for this wrapped new model, previous loss function like *Mean Squared Error(MSE)* doesn't work well. 
The alternative for this loss function is called, **"Cross Entropy Loss"**.

![equation](https://latex.codecogs.com/gif.latex?loss%20%3D%20-%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3Di%7D%5E%7BN%7Dy_%7Bn%7Dlog%5Chat%7By%7D%20&plus;%20%281-y_%7Bn%7D%29log%281-%5Chat%7By_%7Bn%7D%7D%29)

If prediction is correct, loss becomes small and if prediction is wrong, loss becomes large.



#### Sigmoid in Pytorch

```python
# torch.nn.functional.sigmoid(input) : Applies the element-wise function f(x) = 1/(1+exp(-x))

import torch.nn.functional as F

class Model(torch.nn.Module):
  
  def __init__(self):
    super(Model, self).__init__()
    self.linear = torch.nn.Linear(1, 1)   # One in and one out

  def forward(self, x):
    y_pred = F.sigmoid(self.linear(x))
    return y_pred
    

# Cross Entropy as loss function
criterion = torch.nn.BCELoss(size_average=True)    

...

```

### Code for Logistic Regression

```python

import torch
from torch.autograd import Variable
import torch.nn.functional as F

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))   # y_data should be only zero or one


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
        # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
hour_var = Variable(torch.Tensor([[1.0]]))
print("predict 1 hour ", 1.0, model(hour_var).data[0][0] > 0.5)  
hour_var = Variable(torch.Tensor([[7.0]]))
print("predict 7 hours", 7.0, model(hour_var).data[0][0] > 0.5)


```




# 0523-deeper_wider.md
# 23/05/18 Daily Report

## Deeper and Wider Model

#### Wider! (increase of input factor)

In reality, we need various kind of input data ; not only one input factor.
For example, in case of predicting victory or defeat of the game, we should regard player's stats, weather condition, etc.. and so on.


In order to include another factor of input data, we should just expand the matrix in widthwise.
example : 
```python
  # one kind of input
  x_data = [1.0, 2.0, 3.0]
  
  # two kinds of input
  x_data = [
    [1.0, 0.1],
    [2.0, 0.8],
    [3.0, 0.7]
  ]

```
As another input factor is added, weight size should be changed either.
Assuming that *W* is equal to weight(coefficient) matrix, *X* as input and *Y* as output,  
in matrix multiplication form  *XW = Y*, *W* 's row size should be equal to column size of *X*.  

With pytorch, in case of two input data and one output data, we can represent like this code.
```python
linear = torch.nn.Linear(2, 1)  # two input and one output
y_pred = linear(x_data)    
```

#### Dipper!

Between input and output layer, there can be a lot of hidden layers. 
In this case, we should aware that size of input and output is fixed, which means that number of hidden layers do not affect on input size and output size.
For instance, let's consider three layered model :

```python
sigmoid = torch.nn.Sigmoid()  # activation function as example

l1 = torch.nn.Linear(2, 4)
l2 = torch.nn.Linear(4, 3)
l3 = torch.nn.Linear(3, 1)

out1 = sigmoid(l1(x_data))
out2 = sigmoid(l2(out1))
y_pred = sigmoid(l3(out2))

```


##### Implementation of Two Layer Model



```python
# -*- coding: utf-8 -*-
import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)   # equal to relu function
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```

##### Implementation of Three-Layered Model with 8 feature


```python
import torch
from torch.autograd import Variable
import numpy as np

xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

print(x_data.data.shape)
print(y_data.data.shape)


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
        # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```


# 0524-datasetloader.md
# 24/05/18 Daily Report

## DatasetLoader


### Terminology in the neural net data
([from stackoverflow](https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks))
- one **epoch** : one forward pass and one backward pass of *all* the training examples

- **batch size** : the number of training examples in one forward/backward pass. The higher the batch size,
the more memory space you'll need.

- number of **iterations** : number of passes, each pass using [batch size] number of examples.
To be clear, one pass = one forward pass + one backward pass(we do not count the forward pass and backward pass as two different passes).

Example: if you have if you have 1000 training examples, and your batch size is 500, 
then it will take 2 iterations to complete 1 epoch.


### Implementation of DatasetLoader

Pytorch provides convenient way to handle dataset, using
```python 
from torch.utils.data import Dataset, DataLoader
```


- Example

```python

# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, read data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # return one item on the index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    # return the data length
    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Run your training process
        print(epoch, i, "inputs", inputs.data, "labels", labels.data)

```

# 0525-overfitting_sol.md
# 18/05/18 Daily Report



## Solution for Overfitting


Overfitting is "the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably."

There are solutions for this.

First of all, we may consider increase of training data. However, this takes a lot of cost, effort and time. 

In addition, as quantity of training data increases, it takes more time to train.  


Now, let's look at practical solutions for this.

### Regularization

Regularization is kind of penalty condition.

Regularization "smoothes" the trained graph in view-point of the graph.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/regularization.PNG" width="700" height="400">
 

#### Mathematical Representation 

* L2 Regularization
  L2 regularization can be represented in this equation.
  
  ![equation](https://latex.codecogs.com/gif.latex?C%20%3D%20C_%7B0%7D%20&plus;%20%5Cfrac%7B%5Clambda%7D%7B2n%7D%20%5Csum_%7Bw%7Dw%5E%7B2%7D)
  
  - C0 : original cost function
  - n : number of training data
  - ![equation](https://latex.codecogs.com/gif.latex?%5Clambda) : regularization variable
  - w : weight
   
  As regularization term is included, training is proceeded not only to minimize C0 but also to minimize w values.
  
  If we compute the newly defined cost function with derivative of w, new w is determined like this.
  
  ![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20w%20%5Crightarrow%20w%20-%20%5Ceta%20%5Cfrac%7B%5Cpartial%20C_%7B0%7D%7D%7B%5Cpartial%20w%7D%20-%20%5Cfrac%7B%5Ceta%20%5Clambda%7D%7Bn%7Dw%20%5Cnewline%20%5Cnewline%20%5Cnewline%20%3D%20%281-%20%5Cfrac%7B%5Ceta%20%5Clambda%7D%7Bn%7D%29w%20-%20%5Ceta%20%5Cfrac%7B%5Cpartial%20C_%7B0%7D%7D%7B%5Cpartial%20w%7D)
 
 
 In the above equation, let's look at the term ![equation](https://latex.codecogs.com/gif.latex?%281-%20%5Cfrac%7B%5Ceta%20%5Clambda%7D%7Bn%7D%29w).
 
 As coefficient of w is less than 1 and bigger than zero, it will be proceeded in w decreasing way.
 
 We call this *"weight decay"*. With this "weight decay", we can prevent certain weight from being too large.
 
 * L1 Regularization
 ![equation](https://latex.codecogs.com/gif.latex?C%3DC_%7B0%7D&plus;%5Cfrac%7B%5Clambda%7D%7Bn%7D%5Csum_%7Bw%7D%5Cleft%20%7C%20w%20%5Cright%20%7C)
 
 In L1 regularization, first-order term is placed instead of 2nd-order term in L2 case.
 
 If we compute the cost function with derivative of w,
 
 ![equation](https://latex.codecogs.com/gif.latex?w%20%5Crightarrow%20w%5E%7B%27%7D%20%3D%20w%20-%20%5Cfrac%7B%5Ceta%20%5Clambda%7D%7Bn%7Dsgn%28w%29%20-%20%5Ceta%20%5Cfrac%7B%5Cpartial%20C_%7B0%7D%7D%7B%5Cpartial%20w%7D)
 
 We can figure out that regularization is performed subtracting a constant depending on sign of w.
 
 
 #### L2 vs L1
 
 In case of L1, regularization is performed by subtracting constant value. So, small value of weights almost converge to zero, and several important weights will be remained.
 
 Therefore, in order to extract several meaningful value, L1 regularization is more effective than L2.
 
 This is appropriate for *sparse model*.
 
 However, as mathmatical representation shows, there are non-differential points, so we should be careful whenever applying gradiendt based learning.
 
 
 
 ### Data augmentation
 
 Data augmentation is about how to increase training data in efficient way.
 
 Let's assume that we want to develop new algorithm to detect cursive letter.
 
  <img src="https://github.com/jwcse/DeepLearning/blob/master/img/cursive_MNIST.PNG" width="400" height="400">
 
 
 In that case, it is difficult to obtain large enough data and enter the data on database.
 
 There are efficient way to generate good training data.
 
 #### Affine Transform
 With affine transform, we can get good enough training dataset.
 
 <img src="https://github.com/jwcse/DeepLearning/blob/master/img/data_aug_affine.PNG" width="700" height="500">
 
 #### Elastic Distortion
 
 Microsoft developed this method to generate effictive training dataset.
 
 As the picture below shows, generate displacement vector in diverse way.
 
 <img src="https://github.com/jwcse/DeepLearning/blob/master/img/elastic_distortion.PNG" width="600" height="400">
 
 
 ### Dropout
 
 As number of hidden layers increase, training performance gets better.
 
 But, when it comes to size of the layer, there's problem of overfitting.
 
 In such a case, dropout can be a good solution.
 
 #### Overview
 
 <img src="https://github.com/jwcse/DeepLearning/blob/master/img/dropout.PNG" width="600" height="400">
 
 As the picture above shows, training is performed with omitting certain neurons.
 
 The training is performed iteratively with selecting dropouts(the omitted neurons) randomly.
 
 #### Effect
 * Voting Effect
 
 When training on diminished layers of neural net, the remaining layer is trained and it can be somewhat overfitted.
 
 But on next mini0batch, another layer is somewhat overfitted as the prior case.
 
 If these procedure is conducted randomly and iteratively, there can be average effect by voting.
 
 In conclusion, there will be similar effect of regularization..
 
 
 * Avoid Co-adaptation
 
 If certain neuron's bias or weight has enormous value, this can bring about bad performance of training.
 
 But when dropout is accompanied in training, we can avoid the worried situation.
 
 
 
 
 
 
 
 

# 0528-cnn.md
# 28/05/18 Daily Report



## CNN Architecture

*CNN(Convolutional Neural Network)* is neural network utilized in image processing , MLP and so on.
The key idea is *convolution*, which means only small portion of image is handled at a time. This small portion is called 'patch',
and whole input image is processed by size of the patch.


### History

CNN was firstly introduced at 1989 by LeCun's paper ; "Backpropagation applied to handwritten zip code recognition."

The paper showed meaningful result, but it was not enough for commoditization.

At 2003, Simard's paper, "Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis" was published.

The paper simplified the model, and it had become cornerstone for broad use.

### Problems on pre-existing MLP

There were problems on pre-existing MLP.
    
    - Enormous number of free parameters 
    
    - Huge network size
    
    - Long time consumption for training

**CNN** was suggested to resolve the above problems.


### Idea of CNN

In human's neural network, there is part called receptive field, which occurs respond to information processing related cell. 

The characteristic of this receptive field is that, external stimulus affects not only on entire area, but on *certain specific area*.



Likewise, in image or media, pixels in certain area have high correlation with surrounding pixels, but the correlation decreases as distance between pixel increases. 

CNN has appeared based on this idea.



### Process of CNN

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/cnn_overview.PNG" width="800" height="250">


### Overview of CNN
    
<img src="https://github.com/jwcse/DeepLearning/blob/master/img/cnn_overview2.PNG" width="800" height="250">



### Convolution

Assuming that there are two functions, one function is reversed and shifted. 

To that function, the other function is multiplied and then integrated.

These process is performed in unit of filter as filter is moved through the image.

(Let's think one function is image, and the other function is filter)

The picture below shows an example of convolution operation on 2-dimension data.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/convolution_ex.PNG" width="600" height="500">




### Characteristics of CNN

* Locality(Local Connectivity)

    CNN utilizes local information as receptive field. Correlation on spatially adjacent signal is extracted using non-linear filter. By using various filters, local featrue can be extracted. After this, through *subsampling* process and iterative filter operation on local feature, global feature can be extracted.


* Shared Weights

    Number of variables needed can be decreased by using identical filter iteratively.

* Pre-processing is not needed


CNN is classifed as locally connected nerual net, and this kind of neural net produces smaller weights compared to fully connected neural net. 
And this enables flexible handling of input image data.

### Layers - Convolutional, Pooling, Feedforward

#### Convolutional Layer - creation of feature map

Within the size of patch, same size of filter(kernel) which contains weights are computed to input image. The computation is done by 
dot product. After computation, window is moved and computation is perfomed again. Each result of computation value is put into element of feature map. 

The window is moved by size of step, called 'stride'. For example, if stride is equal to two, filter is applied jumping two pixels. 

Also, 'padding' can be done to the original image, which means certain value can be added to boundary of the input value array.


The output size should be different based on stride and patch size. 

For example, let's assume that input image size is 32x32x1, filter size is 5x5x1, and stride is 1x1. 
Then, output array size would be 28x28.

In addition, depending on number of filters, output's depth would be different. Each filter's size is same but contains different values.

If we use 6 filters, for instance, the output's depth would be 6.

#### Pooling Layer - subsample

Purpose is to reduce information generated by convolutional layer.

One of the representative way is *max pooling*. With certain filter size and stride, maximum value in the area is put into the element of output.

Another representative way is *average pooling*, which computes average value in window.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/subsampling.PNG" width="500" height="500">



#### Feedforward layer - classification ; fully connected layer

This utilizes features produced by convolution layer and pooling layer to classify.


### Snippet Code in Pytorch

```python
    # convolution layer
    torch.nn.Conv2d(in_channels, out_channels, kernel_size)
    
    # poolying layer
    nn.MaxPool2d(kernel_size)
    
    # Feedforward layer
    nn.Linear(320, 10)  # and then return with softmax(for example)  
```

### Inception module 
We can use various filters in convolution. The concept of inception is to try possible filters and concatenate all as output.

But before processing with each filter, 1x1 convolution is performed and then processed by each filter.

By doing so, we can reduce quantity of computation singnificantly.
 


### Problems with stacking layers

- Vanishing gradients problem
    
- Back propagation kind of gives up...
    
- Degradation problem ; with increased network depth accuracy gets saturated  and then rapidly degrades


Solution : *Deep Residual Learning*(ResNEt) ; by passing connection -> help propagate gradient gradients 

### Famous paper on CNN

#### Lecun - "Gradient-based learning applied to document recognition"
[link](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
#### Krizhevsky - "ImageNet classification with deep convolutional neural network"
[link](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)


# 0601-meeting.md

# 01/06/18 Meeting Note

## Fundamental Explanation of Deep Learning by Professor

#### 1. Segmentation
*ex>* "Do you want to find out fraction is bone?"
We use **segmentation** when we want to figure out particular part or component in the image or video. We say the target as *'object of interest'* and we implement algorithm to extract what we want.

#### 2. Classification
*ex>* "Let's assume that there are hand-written images representing 0 to 10. What should we do in order to let computer match to each number?"
The example above is the representative classification problem, called [*MNIST*](https://github.com/pytorch/examples/tree/master/mnist)
We can categorize cat or dog from the image. Also, classifying genre from the given movie is possible.

#### 3. SuperResolution

ex> "Assuming that there exists video of low resolution. What if we want larger and better resolution of the given video?"
When we enlarge the size, we can use **interpolation** method. And it's like this.
*original*

1|2
-|-
3|4

*converted*

**1**|1.5|**2**|2.5
----|----|----|----
2|2.5|3|3.5
**3**|3.5|**4**|4.5
4|4.5|5|5.5

But after convertion, boundary between objects in the image becomes blurred. Utilizing many images and extracting features, we can make larger and better image.

#### 4. Registration
ex> "What if we want to figure out how the cancer cell changed after operation?"
In the case of detecting change like the example, we can use the method called **Correspondance**. 
With this method, we can correspond each position from one object to the other one.  
Therefore, we can figure out how the size has changed and how the position has moved.

## Study Direction

- Find out various datasets
In deep learning, dataset is important because training is processed by data.
And if we need labeled data, but data is not lebeled, we cannot do our research with that dataset(importance of manipulated data).
(There exists convenient tool [snorkel](https://github.com/HazyResearch/snorkel/), which helps non-professional researcher to utilize the dataset by assisting labeling. 
But of course, labeling by the professionalist's is the best)
It will be good start point for beginner to search on dataset and figure out how researchers utilize and how they research on. 
[MURA](https://stanfordmlgroup.github.io/competitions/mura/), [kaggle](http://kaggle.com/), [국내공공데이터](https://www.data.go.kr/dataset/3072274/fileData.do) provides such a great dataset.

- Challenging on **Classification** will be helpful!
It's relatively easier than other area.

- Figure out the Definition of the problem
Check out what is input, output and significant process.

In conclusion, select own research way, and proceed on the project asking whenever have questions!




# 0625-meeting.md

# 25/06/18 Meeting Note

## Structure of Research Paper

On every monday, we are going to hold a seminar and review on paper.

Each presenter should address on summary of the research paper and present the keywords in the paper.

We aim to read and review the paper considering the structure of the research paper. 

#### 1. Abstract
*"What, Why, How"*
This part propose what kind of research, why it is important and how the process was done.

#### 2. Intro
  The importance and the reason to solve the problem is suggested.
 
#### 3. Related Works
  This part lists how the researchers has done. The author talks about the problems and limits of the previous research.

#### 4. Method
  This part says how this paper accessed to solve the problem.
  
#### 5. Result
  The result of the research is shown.
  
  In good-quality of paper, authors mention on failure among the research and suggestions for improved research in the future.
  
#### 6. Conclusion/Discussion

#### 7. Reference
  
  
  
### Main Journal & Conference 
- [IEEE PAMI](https://ieeexplore.ieee.org/xpl/aboutJournal.jsp?punumber=34)
- [IEEE MI](https://ieeexplore.ieee.org/xpl/aboutJournal.jsp?punumber=42)
- [CVPR](http://openaccess.thecvf.com/CVPR2017.py)(open access 2017)
- [ICCV](http://openaccess.thecvf.com/ICCV2017.py)(open access 2017)
- [ECCV](https://eccv2018.org/papersubmission/author-guidelines/)(2018 author guideline)

### Application for Management of paper
- Endnote
- Mendeley 
  - Cloud storage is provided 
  - Annotating note is possible
  - Recommend related paper to the user.
  - Helps creating bibliography, so it is easy to embed on paper when writing with Latex.
  


# 0627-activation_functions.md
# 27/06/18 Daily Report


## Various kinds of Activation Functions

There are a lot of functions for activation function in neural net. 

Each activation function has own characteristic, and we should choose activation function by 

### 1. Activation Function
- Function that convert weighted sum into output signal in a neuron. The output contains strength of signal, not binary value.

- Provides non-linearity to the neural net and this enables effect of stacked layers.

- Monotonic(not necessarily), so error surface assosiated with single-layer model is guaranteed to be convex.

- Continuously differentiable ; this is desirable property. In backpropagation among training, gradient-based optimization can be applied.


### 2. Sigmoid Function

- Definition
  
  ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-x%7D%7D)
    
    
- Characteristic
  - Zero-centered, converges to zero as x decreases. Counterwise, converges to one as x increases.
  - *Vanishing Gradient Problem* :
    If output value by activation function is really close to zero or one, derivative will be close to zero.
    In backpropagation process of training, weight would not be updated if the derivative value is almost zero, because update is proceeded by chain rule.
  - Activation value(output) is always over zero
  - Computation of e^x  -> expensive 
  
  
  ### 3. Tanh (Hyperbolic Tangent)
  
  - Definition
    
    ![equation](https://latex.codecogs.com/gif.latex?tanh%28x%29%20%3D%202*sigmoid%282x%29%20-%201%20%3D%20%5Cfrac%7Be%5E%7Bx%7D-e%5E%7B-x%7D%7D%7Be%5E%7Bx%7D&plus;e%5E%7B-x%7D%7D)

  - Characteristic
    - Output range : **[**-1, 1 **]**
    - Trained faster than sigmoid.
    - Still vanishing gradient problem exists.
    
### 4. Relu (Rectified Linear Unit)
  
  - Definition
    
    ![equation](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20max%280%2Cx%29)
    
  - Characteristic
    - Don't need to worry about vanishing gradient problem ; if x is positive, gradient value is equal to one.
    - Converges faster ; Easy to get derivative value and complexity of computation is low.
    - When x is negative, gradient becomes zero.(Cons)
    
### 5. Leaky Relu
 - Definition
    
    ![equation](https://latex.codecogs.com/gif.latex?f%28x%29%3Dmax%280.01x%2C%20x%29)
    
  - Characteristic
    - Cover cons of Relu.
    - Gradient is 0.01 when x is positive and 1 when positive.
    
    
### 6. ELU (Exponential Linear Units)
  
  - Definition
    
    ![equation](https://latex.codecogs.com/gif.latex?%5Cpar%7B%20%5Cnoindent%20%5Calpha%28e%5E%7Bx%7D-1%29%20%5Chspace%7B1cm%7D%20%28x%3C%3D0%29%20%5C%5C%20x%20%5Chspace%7B2.5cm%7D%20%28x%20%3E%200%29%20%7D)
    
  - Characteristic
    - Include positive aspects of Relu.
    - Computation of e^x  -> expensive 
    
### 7. Maxout
 - Definition
    
    ![equation](https://latex.codecogs.com/gif.latex?max%28w_%7B1%7D%5E%7BT%7Dx&plus;b_%7B1%7D%2C%20w_%7B2%7D%5E%7BT%7Dx&plus;b_%7B2%7D%29)
    
  - Characteristic
    - Generalized version of Relu and Leaky relu.
    - Compared with relu, has twice much of parameters -> expensive.
    
   
#### Graph     
![image](https://github.com/jwcse/DeepLearning/blob/master/img/activation_func_graph.png)

# 0629-basic_probability.md
# 29/06/18 Daily Report

## Basic Conception of Probability

### Marginal Probability
Probability of any single event *occurring unconditioned* on any other events.

Whenever someone asks you whether the weather is going to be rainy or sunny today, you are computing a marginal probability.

### Joint Probability 
Probability of *more than one event occurring simultaneously*. 

If I ask you whether the weather is going to be rainy and the temperature is going to be above a certain number, you are computing a **joint probability**.

### Conditional Probability 
Probability of an event occurring given some events that *you have already observed*. 

When I ask you what’s the probability that today is rainy or sunny given that I noticed the temperature is going to be above 80 degrees, you are computing a **conditional probability**.


-----------------


These three concepts are intricately related to each other. Any **marginal probability** can always be computed in terms of sums(**∑**) of **joint probabilities** by a process called **marginalization**.

Mathematically, this looks like this for two events A and B

##### P(A)=∑bP(A,b)

And **joint probabilities** can always be expressed as the product of **marginal probabilities** and **conditional probabilities** by the chain rule.

##### P(A,B)=P(A)⋅P(B|A)

These rules can be generalized to any number of events and are immensely useful in all type of statistical analysis.


##### Reference :
[quora answered by Nipun Ramakrishnan](https://www.quora.com/What-are-marginal-probability-and-conditional-probability)

# 0629-meeting.md
# 29/06/18 Meeting Note

## Meeting with Professor
All the undergraduate research students met professor, and discussed about what is being done and what to do among the vacation.
Based on each student's plan and suggestion, task was allocated.

- Me :
  As graduation is planned on next year February, development of practical software rather than research was allocated.
  -> Utilizing deep learning library or optimized weights, development of android application such that capture object and classify or categorize like that.   
  

# 0702-bayes_generative.md
# 02/07/18 Daily Report

## Bayesian Theorem

The **Bayessian Theorem** is theory which comes from paper *'Essay towards solving a problem in the doctrine of chances'* published by Thomas Bayes. 

In conditional probability, the probability is improved whenever new information is acquired and provided. 
The initial probability before improvement is called *prior probability* and the improved one is called 'posterior probability'.

This **Bayes' theorem** is the theory which contains this improvement.


Let's assume that we want to classify something. 

Let's say variable is *x*, class that we want to classify *C*, the probability for *x* as *P(x)*, 

probabiltiy of arbitrary sample to belong to class *C* as *P(C)*, and conditional probability for *x* in given class *C* as *P(x|C)*.

What we want to get is, the probability of belonging to class *C* with given *x*. We can compute this with this formula.

![equation](https://latex.codecogs.com/gif.latex?P%28C%7Cx%29%20%3D%20%5Cfrac%7BP%28C%29P%28x%7CC%29%7D%7BP%28x%29%7D)


## Generative Model

The viewpoint of model which generates data.

Within this model, we can say that infer the data generating model using data, and classify the given data's class.

The **Bayes' theorem** is used in this process.

The oppositite concept is **Discriminatie model**, which does not concern about how the data is generated and focus on just classifying.

### Generative vs Discriminative
<img src="https://github.com/jwcse/DeepLearning/blob/master/img/generative-discriminative.jpg" width="600" height="500">

# 0703-seminar.md
# 03/07/18 Daily Report

## Seminar

Undergraduate students and several graduate students in Image Lab gathered to hold a seminar.

Each undergraduate students prepared each section with selected paper chosen by individual preferences.

These are papers each students chose. Each students summarized the paper and then explained keywords extracted from the paper.

* Ki-Soo : [Survey on Semantic Image Segmentation Techniques](https://ieeexplore.ieee.org/document/8389420/)

* Jin-Woo(Me) : [Every Picture Tells a Story: Generating Sentences from Images](https://www.researchgate.net/publication/221303952_Every_Picture_Tells_a_Story_Generating_Sentences_from_Images)

* Jin-Woo(Jeon) : [Fully Convolutional Networks for Semantic Segmentation](https://ieeexplore.ieee.org/document/7478072/)

* Yeon-Ji : [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981)

# 0704-PGM.md
# 04/07/18 Daily Report

## PGM(Probabilistic Graphical Model)

### Why we should learn?
  
    To deal with the uncertainty in reality by modeling the world in the form of a probability distribution.   

### Intro
  
  Probabilistic Graphical models are a marriage between graph theory and probability theory.
  They clarify the relationship between neural networks and related network-based models such asHMMs, MRFs, and Kalman filters.
  Probabilistic modeling is widely used throughout machine learning and in many real-world applications. 
  These techniques can be used to solve problems in fields as diverse as medicine, language processing, vision, and many others.

### Key idea
  - Represent the world as a collection of random variables *X1, . . . , Xn* with joint distribution *p(X1, . . . , Xn)*
  - Learn the distribution from data
  - Perform “inference” (compute conditional distributions *p(Xi| X1 = x1, . . . , Xm = xm)*)


### Advantages of the graphical model point of view
  - Incorporation of domain knowledge and causal (logical) structures
  - Inference and learning are treated together
  - Supervised and unsupervised learning are merged seamlessly
  - Missing data handled nicely
  - A focus on conditional independence and computational issues
  - Interpretability (if desired)
  
### Two main kinds of PGM
  - Directed graphical model(DGM)
  - Undirected graphical model(UGM)
  
### Learning and Inference
  - It is not necessary to learn that which can be inferred
  - The weights in a network make local assertions about the relationships between neighboring nodes
  - Inference algorithms turn these local assertions into global assertions about the relationships between nodes 
    - e.x) the probability of an input vector given an output vector
  - This is achieved by associating a joint probability distribution with the network

### Two important rules
  - **Chain rule**
    Let S1, . . . Sn be events, p(Si) > 0
    
    ![equation](https://latex.codecogs.com/gif.latex?p%28S_%7B1%7D%20%5Ccap%20S_%7B2%7D%20%5Ccap%2C%20...%2C%5Ccap%20S_%7Bn%7D%29%20%3D%20p%28S_%7B1%7D%29p%28S_%7B2%7D%20%7C%20S_%7B1%7D%29...%20p%28S_%7Bn%7D%20%7C%20S_%7B1%7D%2C%20.%20.%20.%20%2C%20S_%7Bn-1%7D%29)
  - **Bayes’ rule** 
    Let C, x be events, p(C) > 0 and p(x) > 0
    
    ![equation](https://latex.codecogs.com/gif.latex?P%28C%7Cx%29%20%3D%20%5Cfrac%7BP%28C%29P%28x%7CC%29%7D%7BP%28x%29%7D)


### Representation of PGM
  
<img src="https://github.com/jwcse/DeepLearning/blob/master/img/pgm_representation.png" width="700" height="500">



#### Reference :
[Stanford CS 228: Probabilistic Graphical Models - preliminaries/introduction](https://ermongroup.github.io/cs228-notes/preliminaries/introduction/)

[Introduction to Graphical Models by Michael I. Jordan](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.7467&rep=rep1&type=pdf)

[Medium posting by Neeraj Sharma : Understanding Probabilistic Graphical Models Intuitively](https://medium.com/@neerajsharma_28983/intuitive-guide-to-probability-graphical-models-be81150da7a)

# 0705-independence__BaysNet.md
# 05/07/18 Daily Report


## Independence and Graphical Model

### Independence
  Two events are called **independent** if and only if P(A∩B)=P(A)P(B) (or equivalently, P(A∣B)=P(A)). And this can be denoted as A ⊥ B.
  The independence is equivalent to saying that observing B does not have any effect on the probability of A.
  

### Graphs & Independent Sets
A graph 𝐺 = (𝑉, 𝐸) is defined by a set of vertices 𝑉 and a set of edges 𝐸 ⊆ 𝑉 × 𝑉 (i.e., edges correspond to pairs of vertices).

A set 𝑆 ⊆ 𝑉 is an independent set if there does not exist an edge in 𝐸 joining any pair of vertices in 𝑆.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/independent_set_ex.PNG" width="600" height="900">


### Representation of Independence Structure with Graphical Model 
The amount of storage and the complexity of statistical inference are both affected by the independence structure of the joint probability distribution
  - **More independence means easier computation and less storage**
  - Want models that somehow make the underlying independence assumptions explicit, so we can take advantage of them
  (expensive to check all of the possible independencerelationships)
  - The simple, still powerful model!



## DGM(Directed Graphical Model) a.k.a Bayesian Networks
  
### Definition
A family of probability distributions that admit a compact parametrization that can be naturally described using a directed graph.
A Bayesian network is a directed graphical model that represents independence relationships of a given probability distribution.

* Directed acyclic graph (DAG), 𝐺 = (𝑉, 𝐸)
    - Edges are still pairs of vertices, but the edges (1,2) and (2,1) are now distinct in this model
* One node for each random variable
* One conditional probability distribution per node
* Directed edge represents a direct statistical dependence
* Corresponds to a factorization of the joint distribution

    ![equation](https://latex.codecogs.com/gif.latex?p%28x_%7B1%7D%2C...%2Cx_%7Bn%7D%29%20%3D%20%5Cprod_%7Bi%7Dp%28x_%7Bi%7D%7Cx_%7Bparents%28i%29%7D%29)

### Example
As an example, consider a model of a student’s grade g on an exam; this grade depends on several factors: 
the exam’s difficulty d, the student’s intelligence i, his SAT score s; it also affects the quality l of the reference letter from the professor who taught the course. 

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/bays_net_ex.png" width="750" height="600">

Each variable is binary, except for g, which takes 3 possible values.
Bayes net model describing the performance of a student on an exam. The distribution can be represented a product of conditional probability distributions specified by tables. 
The form of these distributions is described by edges in the graph. The joint probability distribution over the 5 variables naturally factorizes as follows

  ![equation](https://latex.codecogs.com/gif.latex?p%28l%2Cg%2Ci%2Cd%2Cs%29%3Dp%28l%7Cg%29p%28g%7Ci%2Cd%29p%28i%29p%28d%29p%28s%7Ci%29)

The graph clearly indicates that the letter depends on the grade, which in turn depends on the student’s intelligence and the difficulty of the exam.


### Bayesian networks are generative models
In the above example, to determine the quality of the reference letter, we may first sample an intelligence level and an exam difficulty; 
then, a student’s grade is sampled given these parameters; finally, the recommendation letter is *generated* based on that grade.


### Independencies described by Directed Graphs
Independencies can be recovered from the graph by looking at three types of structures.
 <img src="https://github.com/jwcse/DeepLearning/blob/master/img/three_types_components_of_bayes.PNG" width="830" height="600">
  * Common parent
  
    If G is of the form A←B→C, and B is observed, then A⊥C∣B. 
    
    However, if B is unobserved, then A⊥̸C. Intuitively this stems from the fact that B contains all the information that determines the outcomes of A and C
  * Cascade
    
    If G equals A→B→C, and B is again observed, then, again ​A⊥C∣B. However, if B is unobserved, then A⊥̸C. 
    Here, the intuition is again that B holds all the information that determines the outcome of C;
    thus, it does not matter what value A takes
  
  * V-structure
  
    If G is A→C←B, then knowing C couples A and B. In other words, A⊥B if C is unobserved, but A⊥̸B∣C if C is observed.
    
These structures clearly describe the independencies encoded by a three-variable Bayesian net. We can extend them to general networks by applying them recursively over any larger graph. This leads to a notion called **d-separation** (where d stands for directed).

We say that Q, W are d-separated when variables O are observed if they are not connected by an active path. An undirected path in the Bayesian Network structure G is called *active* given observed variables O if for every consecutive triple of variables X,Y,Z on the path, one of the following holds:
  - X←Y←Z, and Y is unobserved Y∉O
  - X→Y→Z, and Y is unobserved Y∉O
  - X←Y→Z, and Y is unobserved Y∉O
  - X→Y←Z, and Y or any of its descendents are observed.
  
 The notion of d-separation is useful, because it lets us describe a large fraction of the dependencies that hold in our model.

#### Reference :
[Stanford CS 228: Probabilistic Graphical Models - Bays Net](https://ermongroup.github.io/cs228-notes/representation/directed/)

[UTD slides](http://www.utdallas.edu/~nrr150130/cs6347/2016sp/lects/Lecture_2_Bayes.pdf)

# 0706-Markov.md
# 06/07/18 Daily Report

## UGM(Unirected Graphical Model) a.k.a MRF(Markov Random Fields)

### Intro - Necessity of UGM
Bayesian networks are a class of models that can compactly represent many interesting probability distributions.
However, some distributions cannot be perfectly represented by a Bayesian network.
  
In such cases, unless we want to introduce false independencies among the variables of our model, we must fall back to a less compact representation (which can be viewed as a graph with additional, unnecessary edges). 
This leads to extra, unnecessary parameters in the model, and makes it more difficult to learn these parameters and to make predictions.  

There exists, however, another technique for compactly representing and visualizing a probability distribution that is based on the language of undirected graphs. 
This class of models (known as Markov Random Fields or MRFs) can compactly represent distributions that directed models cannot represent.   
  
  
 ### Definition
 A Markov Random Field (MRF) is a probability distribution p over variables x1,...,xn defined by an undirected graph G in which nodes correspond to variables xi. 
 The probability p has the form
 
 ![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20p%28x_%7B1%7D%2C...%2Cx_%7Bn%7D%29%3D%5Cfrac%7B1%7D%7BZ%7D%5Cprod_%7Bc%5Cepsilon%20C%7D%20%5Cphi%20_%7Bc%7D%20%28x_%7Bc%7D%29%20%5Cnewline%5Cnewline%5Cnewline%20Z%20%3D%20%5Csum_%7Bx_%7B1%7D%2C...%2Cx_%7Bn%7D%7D%20%5Cprod_%7Bc%5Cepsilon%20C%7D%20%5Cphi%20_%7Bc%7D%20%28x_%7Bc%7D%29)
 
 where C denotes the set of cliques (i.e. fully connected subgraphs) of G. 
  * clique : clique (/ˈkliːk/ or /ˈklɪk/) is a subset of vertices of an undirected graph such that every two distinct vertices in the clique are adjacent; that is, its induced subgraph is complete. Cliques are one of the basic concepts of graph theory and are used in many other mathematical problems and constructions on graphs. 
 
 The value is a normalizing constant(a.k.a partition function) that ensures that the distribution sums to one.
 
 
### Example
As a motivating example, suppose that we are modeling voting preferences among persons A,B,C,D. Let’s say that (A,B), (B,C), (C,D), and (D,A) are friends, and friends tend to have similar voting preferences. 
These influences can be naturally represented by an undirected graph.

One way to define a probability over the joint voting decision of A,B,C,D is to assign scores to each assignment to these variables and then define a probability as a normalized score. A score can be any function, but in our case, we will define it to be of the form

![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20%5Ctilde%7Bp%7D%28A%2CB%2CC%2CD%29%20%3D%20%5Cphi%28A%2CB%29%5Cphi%28B%2CC%29%5Cphi%28C%2CD%29%5Cphi%28D%2CA%29%20%5Cnewline%20p%28A%2CB%2CC%2CD%29%3D%5Cfrac%7B1%7D%7BZ%7D%5Ctilde%7Bp%7D%28A%2CB%2CC%2CD%29)

And we can represent in graph like this

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/markvov_ex.PNG" width="600" height="400">
 
 
### Feature
Note that unlike in the directed case, we are not saying anything about how one variable is generated from another set of variables (as a conditional probability distribution would do). 
We simply indicate a level of coupling between dependent variables in the graph.
In a sense, this requires less prior knowledge, as we no longer have to specify a full generative story of how the vote of B is constructed from the vote of A (which we would need to do if we had a P(B∣A) factor). 
Instead, we simply identify dependent variables and define the strength of their interactions;
this in turn defines an energy landscape over the space of possible assignments and we convert this energy to a probability via the normalization constant.

  * Advantage
    - They can be applied to a wider range of problems in which there is no natural directionality associated with variable dependencies.

    - Undirected graphs can succinctly express certain dependencies that Bayesian nets cannot easily describe (although the converse is also true)
  
  * Disadvantage
    - Computing the normalization constant Z requires summing over a potentially exponential number of assignments. Can be NP-hard; thus many undirected models will be intractable and will require approximation techniques.
    
    - Undirected models may be difficult to interpret.
    
    - It is much easier to generate data from a Bayesian network, which is important in some applications.
  
It is not hard to see that Bayesian networks are a special case of MRFs with a very specific type of clique factor (one that corresponds to a conditional probability distribution and implies a directed acyclic structure in the graph), 
and a normalizing constant of one. In particular, if we take a directed graph G and add side edges to all parents of a given node (and removing their directionality), then the CPDs (seen as factors over a variable and its ancestors) factorize over the resulting undirected graph. 
The resulting process is called *moralization*.


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/moralization_ex.png" width="500" height="400">


The converse is also possible, but may be computationally intractable, and may produce a very large (e.g. fully connected) directed graph.
  
Thus, MRFs have more power than Bayesian networks, but are more difficult to deal with computationally. 
A general rule of thumb is to use Bayesian networks whenever possible, and only switch to MRFs if there is no natural way to model the problem with a directed graph (like in our voting example).  
 



#### Reference :
[Stanford CS 228: Probabilistic Graphical Models - MRF](https://ermongroup.github.io/cs228-notes/representation/undirected/)

[UTD slides](http://www.utdallas.edu/~nrr150130/cs6347/2016sp/lects/Lecture_4_MRFs.pdf)

# 0709-seminar.md
# 09/07/18 Daily Report

## Seminar - 2nd

Undergraduate students and several graduate students in Image Lab gathered to hold a seminar.

Each undergraduate student prepared own section with selected paper chosen by individual preferences.

In addition, each student complemented last week's presentation by studying more on mis-understood part on last week. 


* Ki-Soo : Surveyed on all the **metrics** used on research. 
  - The term **metric** is method to calculate similarity between two objects.

  - When it comes to clustering, data is grouped into N group(N defined by input). The data in the same cluster shares common feature, and this feature is based on the similarity.

  - Euclidean distance, Manhattan distance, Minkowski distance, Edit distance, Chebyshev distance, Cosine similarity... etc.


* Jin-Woo(Me) : Surveyed on **MRF**
  - Introduction of **PGM(Probabilistic Graphical Models)**
  - Concept of **Bayseian Network**
  - Concept of **MRF(Markov Random Field)**
  
  
* Jin-Woo(Jeon) : Presented on [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
  - On last week seminar, there was a question on weak point of CNN. To clarify the weak point and find out the alternative, researched on **CapsNet**.
  - **CapsNet** was proposed by Geoffrey Hinton and used **Dynamic Routing** for solution.  


* Yeon-Ji : Presented on [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  - To complement last week's keyword, researched on **Batch Normalization**

# 0712-simulated_annealing.md
# 12/07/18 Daily Report


## Simulated Annealing

### What is Simulated Annealing
The simulated annealing algorithm was originally inspired from the process of annealing in metal work. 

Annealing involves heating and cooling a material to alter its physical properties due to the changes in its internal structure. As the metal cools its new structure becomes fixed,
consequently causing the metal to retain its newly obtained properties. 
In simulated annealing we keep a temperature variable to simulate this heating process. 
We initially set it high and then allow it to slowly 'cool' as the algorithm runs. 
While this temperature variable is high the algorithm will be allowed, with more frequency, to accept solutions that are worse than our current solution. This gives the algorithm the ability to jump out of any local optimums it finds itself in early on in execution. As the temperature is reduced so is the chance of accepting worse solutions, therefore allowing the algorithm to gradually focus in on a area of the search space in which hopefully, a close to optimum solution can be found. This gradual 'cooling' process is what makes the simulated annealing algorithm remarkably effective at finding a close to optimum solution when dealing with large problems which contain numerous local optimums.
The nature of the traveling salesman problem makes it a perfect example.

### Advantages 

You may be wondering if there is any real advantage to implementing simulated annealing over something like a simple hill climber. Although hill climbers can be surprisingly effective at finding a good solution, they also have a tendency to get stuck in local optimums. As we previously determined, the simulated annealing algorithm is excellent at avoiding this problem and is much better on average at finding an approximate global optimum.

To help better understand let's quickly take a look at why a basic hill climbing algorithm is so prone to getting caught in local optimums.

A hill climber algorithm will simply accept neighbour solutions that are better than the current solution. When the hill climber can't find any better neighbours, it stops.

To compare with gradient descent, gradient descent works only if we have a good initial segmentation. However, simulated annealing always works(at least in theory).



### Algorithm


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/sim_anneal_alg_structure.jpg" width="420" height="600">
    
### Reference

[simulated-annealing-algorithm-for-beginners](http://www.theprojectspot.com/tutorial-post/simulated-annealing-algorithm-for-beginners/6)
    

# 0713-MRF_denoise_code.md
# 13/07/18 Daily Report


## Denoising using MRF 

The code has originated from [andredung's code](https://github.com/andreydung/MRF) which is based on *MRF* and 
*SNR(Signal to Noise Ratio)* algorithm.

### Code
```python

# Image denoising using MRF model
from PIL import Image
import numpy
from pylab import *

def main():
	# Read in image
	im=Image.open('lena512.bmp')
	im=numpy.array(im)
	im=where (im>100, 1, 0) #convert to binary image
	(M,N)=im.shape

	# Add noise
	noisy_img=im.copy()
	noise=numpy.random.rand(M,N)	# generate M x N size with random value 0 ~ 1
	ind=where(noise<0.2)	# get index list which satisfies the condition
	noisy_img[ind]=1-noisy_img[ind]	# manipulate noise image

	# show noisy image
	gray()
	title('Noisy Image')
	imshow(noisy_img)

	# process by MRF de-noise
	out=MRF_denoise(noisy_img)

	# show de-noised image
	figure()		
	gray()
	title('Denoised Image')
	imshow(out)
	show()

def MRF_denoise(noisy_img):
	# Start MRF	
	(M,N) = noisy_img.shape
	y_old = noisy_img
	y = zeros((M,N))

	while SNR(y_old, y)>0.01:
		print(SNR(y_old,y))
		for i in range(M):
			for j in range(N):
				index=neighbor(i,j,M,N)
				
				a=cost(1, noisy_img[i, j], y_old, index)
				b=cost(0, noisy_img[i, j], y_old, index)

				if a>b:
					y[i,j]=1
				else:
					y[i,j]=0
		y_old=y
	print(SNR(y_old,y))
	return y


def SNR(A,B):
	if A.shape==B.shape:
		return numpy.sum(numpy.abs(A-B))/A.size
	else:	# Exception case
		raise Exception("Two matrices must have the same size!")


def delta(a,b):
	return 1 if a== b else 0


def neighbor(i,j,M,N):
	"""
		i : row index
		j : col index
		M : image's row size
		N : image's col size
		:return  neighboring points
	"""

	# find correct neighbors
	if i==0 and j==0:	# top-left corner
		neighbor=[(0,1), (1,0)]
	elif i==0 and j==N-1:	# top-right corner
		neighbor=[(0,N-2), (1,N-1)]
	elif i==M-1 and j==0:	# bottom-left corner
		neighbor=[(M-1,1), (M-2,0)]
	elif i==M-1 and j==N-1:		# right-bottom corner
		neighbor=[(M-1,N-2), (M-2,N-1)]
	elif i==0:		# first row
		neighbor=[(0,j-1), (0,j+1), (1,j)]
	elif i==M-1:	# last row
		neighbor=[(M-1,j-1), (M-1,j+1), (M-2,j)]
	elif j==0:		# first column
		neighbor=[(i-1,0), (i+1,0), (i,1)]
	elif j==N-1:	# last column
		neighbor=[(i-1,N-1), (i+1,N-1), (i,N-2)]
	else:			# inside boundary
		neighbor=[(i-1,j), (i+1,j), (i,j-1), (i,j+1),\
				  (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]

	return neighbor


def cost(y,x,y_old,index):
	alpha=1
	beta=10
	return alpha*delta(y,x)+\
		beta*sum(delta(y,y_old[i]) for i in index)


if __name__=="__main__":
	main()

```

### Result
<img src="https://github.com/jwcse/DeepLearning/blob/master/img/mrf_denoise_ex_result.PNG" width="700" height="420">

# 0713-MRF_segmentation.md
# 13/07/18 Daily Report


## Segmentation using MRF 

The code has originated from [tarunz's code](https://github.com/tarunz/Image-Segmentation-MRF/) which is based on *MRF* and 
*simulated annealing* algorithm.

### Code
```python

# coding: utf-8


import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from math import *



imagepath = 'Images\scar.jpg'
SEGS = 2
NEIGHBORS = [(-1,0) , (1,0) , (0,-1) , (0,1)]
BETA = 1
TEMPERATURE = 4.0
ITERATIONS = 1000000
COOLRATE = 0.95



def isSafe(M, N, x, y):
    return x>=0 and x<M and y>=0 and y<N

def delta(i,l):
    if i==l:
        return -BETA
    return BETA


# In[4]:

def reconstruct(labs):
    labels = labs
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            labels[i][j] = (labels[i][j]*255)/(SEGS-1)
    return labels



def calculateEnergy(img, variances, labels):
    M,N = img.shape
    energy = 0.0
    for i in range(M):
        for j in range(N):
            l = labels[i][j]
            energy += log(sqrt(variances[l]))
            for (p,q) in NEIGHBORS:
                if isSafe(M, N, i+p, j+q):
                    energy += (delta(l,labels[i+p][j+q])/2.0)
    return energy



def variance(sums1,squares1,nos1):
    return squares1/nos1-(sums1/nos1)**2



def initialize(img):
    labels = np.zeros(shape=img.shape ,dtype=np.uint8)
    nos = [0.0]*SEGS
    sums = [0.0]*SEGS
    squares = [0.0]*SEGS
    for i in range(len(img)):
        for j in range(len(img[0])):
            l = randint(0,SEGS-1)
            sums[l] += img[i][j]
            squares[l] += img[i][j]**2
            nos[l] += 1.0
            labels[i][j] = l
    return (sums, squares, nos, labels)

def run():
    # read image
    original = cv2.imread(imagepath)

    # convert to binary
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # initialize
    sums, squares, nos, labels = initialize(img)
    variances = [variance(sums[i], squares[i], nos[i]) for i in range(SEGS)]

    energy = calculateEnergy(img, variances, labels)

    temp = TEMPERATURE
    it = ITERATIONS
    while it > 0:
        (M, N) = img.shape
        change = False
        # select random one pixel
        x = randint(0, M - 1)
        y = randint(0, N - 1)
        val = float(img[x][y])  # intensity
        l = labels[x][y]
        newl = l
        while newl == l:
            newl = randint(0, SEGS - 1)

        val = float(val)

        remsums = sums[l] - val
        addsums = sums[newl] + val

        remsquares = squares[l] - val * val
        addsquares = squares[newl] + val * val

        remvar = variance(remsums, remsquares, nos[l] - 1)
        addvar = variance(addsums, addsquares, nos[newl] + 1)

        newenergy = energy

        newenergy -= log(sqrt(variance(sums[l], squares[l], nos[l]))) * (nos[l])
        newenergy += log(sqrt(remvar)) * (nos[l] - 1)

        newenergy -= log(sqrt(variance(sums[newl], squares[newl], nos[newl]))) * (nos[newl])
        newenergy += log(sqrt(addvar)) * (nos[newl] + 1)

        # process by neighbors
        for (p, q) in NEIGHBORS:
            if isSafe(M, N, x + p, y + q):
                newenergy -= delta(l, labels[x + p][y + q])
                newenergy += delta(newl, labels[x + p][y + q])

        # should we update or not?
        if newenergy < energy:
            change = True
        else:
            prob = 1.1
            if temp != 0:
                prob = np.exp((energy - newenergy) / temp)
            if prob >= (randint(0, 1000) + 0.0) / 1000:
                change = True

        if change:
            labels[x][y] = newl
            energy = newenergy

            nos[l] -= 1
            sums[l] = remsums
            squares[l] = remsquares

            nos[newl] += 1
            sums[newl] = addsums
            squares[newl] = addsquares

        temp *= COOLRATE
        it -= 1

    plt.imshow(reconstruct(labels), interpolation='nearest', cmap='gray')
    plt.imshow(img, cmap='gray')
    cv2.imwrite("segmented.jpg", labels)


if __name__ == "__main__":
    run()



```

### Result
<img src="https://github.com/jwcse/DeepLearning/blob/master/img/mrf_seg_ex_result.PNG" width="800" height="400">

# 0716-seminar.md
# 16/07/18 Daily Report

## Seminar

Undergraduate students and several graduate students in Image Lab gathered to hold a seminar.

Each undergraduate student prepared own section with selected paper chosen by individual preferences.

In addition, each student complemented last week's presentation by studying more on mis-understood part on last week. 

### CNN
Ki-Soo surveyed on CNN(Convolutional Neural Network). 

#### Intro
CNN is a kind of deep learning.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/cnn_inspiration.PNG" width="500" height="500">

The picture above indicates research by Hubel and Wiesel.

This shows observation of responding neuron when animal watches a certain object.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/cnn_inspiration2.PNG" width="500" height="500">

If we take a look at the result of observation, we can figure out that neurons are reacting on certain part of the object(not all of image).

Based on this inspiration, CNN was invented.

CNN shows good performance on various fields like image and text.

#### CNN experiment on MNIST
```python
    import tensorflow as tf
    import random

    from tensorflow.examples.tutorials.mnist import input_data

    tf.set_random_seed(777)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

    keep_prob = tf.placeholder(tf.float32)

    X = tf.placeholder(tf.float32, [None, 784])
    X_img = tf.reshape(X, [-1, 28, 28, 1]) # img 28 * 28 * 1
    Y = tf.placeholder(tf.float32, [None, 10])

    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    '''
    Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
    Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
    Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
    Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
    '''

    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    '''
    Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
    Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
    Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
    Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
    '''

    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

    '''
    Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
    Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
    Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
    Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
    Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
    '''

    W4 = tf.get_variable(name='9', shape=[128 * 4 * 4, 625],
                        initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([625]))
    L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

    '''
    Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
    Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
    '''

    W5 = tf.get_variable(name='0', shape=[625, 10],
                        initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([10]))
    logits = tf.matmul(L4, W5) + b5

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={ X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))
```


And this is the result.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/cnn_seminar_result.PNG" width="200" height="400">


### Decaying LR vs Batch Normalization
Jin-Woo(Jeon) surveyed on decaying learning rate and batch normalization.

#### Implementation of MRF 
Presented on segmentation and denoising using MRF method.
  

# 0717-android_ML.md
# 17/07/18 Daily Report


## Machine Learning on Android

### Necessity of ML on Android
1. UX
  Regarding response speed of service and case of offline situation, on-device machine learning is necessary.

2. Cost
  - Battery consumption cost
  - Data network consumption ; whenever request on server and case of uploading large size of data. 

3. Privacy
  Whenever user does not want to provide own data to ML platform.
  
  
### Ways of ML on Android
There are three ways to enable machine learning on Android platform.

#### Using JNI to bridge into the NDK

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/JNI_NDK.png" width="700" height="500">

  * NDK : Native Development Kit
    NDK connects App(Java) and Library(C/C++) using JNI interface. We say "native method" for implementation using C/C++ in JVM.
    JDK supports JNI, so calling C/C++ code in JVM is possible.

This code is complex and hard to maintain, e.g. the JNI code needs to be built differently from normal Android Studio/Gradle builds.


#### TensorFlowInferenceInterface class

To make this easier, in late 2016 Google added the TensorFlowInferenceInterface class (GitHub commits). 
This helped standardize how to interface with TensorFlow models from Java. 
It provides these prebuilt libraries:

  - libandroid_tensorflow_inference_java.jar — the Java interface layer.
  - libtensorflow_inference.so — the JNI code that talks to the TensorFlow model.
  
The picture below is screenshot of one example. 
  
<img src="https://github.com/jwcse/DeepLearning/blob/master/img/TensorFlowInferenceInterface_ex.png" width="600" height="400">



#### More Simple Using TensorFlowInferenceInterface

Simply by adding a dependency to your build.gradle and then using the TensorFlowInferenceInterface class, it has been much easier.

What we have to do is:
  1.Include the compile 'org.tensorflow:tensorflow-android:+' dependency in your build.gradle.
  
  2. Use the TensorFlowInferenceInterface to interface with your model.


### Efficient TensorFlow model for Mobile : TensorFlow Lite

TensorFlow Lite is TensorFlow’s lightweight solution for mobile and embedded devices. It lets you run machine-learned models on mobile devices with low latency, so you can take advantage of them to do classification, regression or anything else you might want without necessarily incurring a round trip to a server.

TensorFlow Lite is comprised of a runtime on which you can run pre-existing models, and a suite of tools that you can use to prepare your models for use on mobile and embedded devices.

It’s not yet designed for training models. Instead, you train a model on a higher powered machine, and then convert that model to the .TFLITE format, from which it is loaded into a mobile interpreter.


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/tensorflowlite.png" width="450" height="406">

# 0723-seminar.md
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



# 0730-seminar.md
# 30/07/18 Daily Report

## Seminar

Undergraduate students and several graduate students in Image Lab gathered to hold a seminar.

Each undergraduate student prepared own section with selected paper chosen by individual preferences.

In addition, each student complemented last week's presentation by studying more on mis-understood part on last week. 

These are topics covered in the seminar.

## Derivation of Gradient Descent
We want to minimize a convex, continuous and differentiable loss function ℓ(w).

![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20Initialize%20%5Chspace%7B5%7D%20%5Coverrightarrow%7Bw_%7B0%7D%7D%20%5Cnewline%20Repeat%20%5Chspace%7B5%7Duntil%5Chspace%7B5%7D%20converge%20%3A%20%5Cnewline%20%5Cindent%20%5Coverrightarrow%7Bw_%7Bt&plus;1%7D%7D%20%3D%20%5Coverrightarrow%7Bw_%7Bt%7D%7D&plus;%5Coverrightarrow%7Bs%7D%20%5Cnewline%20%5Cindent%20If%20%5Cleft%20%5C%7C%20%5Coverrightarrow%7Bw_%7Bt&plus;1%7D%7D%3D%5Coverrightarrow%7Bw_%7Bt%7D%7D%20%5Cright%20%5C%7C_%7B2%7D%20%3C%20%5Cvarepsilon%20%2C%20%5Chspace%7B5%7D%20converged%21)

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/seminar_gradient_descent.png" width="400" height="300">

### Trick - Taylor Expansion

How can you minimize a function ℓ if you don't know much about it? 
The trick is to assume it is much simpler than it really is. 
This can be done with Taylor's approximation.
Provided that the norm ![equation](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7C%20s%20%5Cright%20%5C%7C_%7B2%7D)]is small,
we can approximate the function ![equation](https://latex.codecogs.com/gif.latex?l%28%5Cvec%7Bw%7D&plus;%5Cvec%7Bs%7D%29) by its first and second derivatives: 

![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20l%28%5Cvec%7Bw%7D&plus;%5Cvec%7Bs%7D%29%5Capprox%20l%28%5Cvec%7Bw%7D%29&plus;%5Cbigtriangledown%20l%28%5Cvec%7Bw%7D%29%5E%7BT%7D%5Cvec%7Bs%7D%5Cnewline)


Our goal is to **find a vector ![equation](https://latex.codecogs.com/gif.latex?%5Cvec%7Bs%7D) that minimizes this function**. In steepest descent we simply set 

![equation](https://latex.codecogs.com/gif.latex?%5Cvec%7Bs%7D%20%3D%20-%5Calpha%5Cbigtriangledown%20l%28%5Cvec%7Bw%7D%29)

for some small α>0. 

It is straight-forward to prove that in this case ![equation](https://latex.codecogs.com/gif.latex?l%28%5Cvec%7Bw%7D&plus;%5Cvec%7Bs%7D%29%20%3C%20l%28%5Cvec%7Bw%7D%29). 

![equation](https://latex.codecogs.com/gif.latex?l%28%5Cvec%7Bw%7D&plus;%28-%5Calpha%5Cbigtriangledown%20l%28%5Cvec%7Bw%7D%29%29%20%5Capprox%20l%28%5Cvec%7Bw%7D%29%20-%20%5Calpha%5Cbigtriangledown%20l%28%5Cvec%7Bw%7D%29%5E%7BT%7D%5Cbigtriangledown%20l%28%5Cvec%7Bw%7D%29%20%3C%20l%28%5Cvec%7Bw%7D%29)

By this way, newly updated value would be smaller than original value.

Setting learning rate(alpha) is kind of dart art that this too small value will bring about small performance. On the other hand, in case of large value or learning rate, the algorithm can easily diverge out of control.


#### Reference
[Gradient Descent (and Beyond) from Cornell education materials](http://www.cs.cornell.edu/courses/cs4780/2015fa/web/lecturenotes/lecturenote07.html)

# 0731-dimension_reduction.md
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


# 0801-PCA_LDA.md
# 01/08/18 Daily Report


## PCA(Principal Component Analysis)

### Basic Knowledge

* **Variance** : 
  Represents "how much the data is far from the average"

* **Covariance** : 
  Represents "relationship between group of data points"
  
### Definition
A way of linear dimensional reduction which aims to figure out the axis representing the dataset best.

The *axis* is equal to 

  = Spreading the data most broadly
  
  = Having the biggest variance

  = **Principal component**
  
  
### Assumption of PCA
1. Submanifold representing data has *linear basis*.

2. The vector of biggest variance contains the most information.

3. Principal components are orthogonal each other.

### Algorithm to get Convertion Matrix U
  * Input : training data ![equation](https://latex.codecogs.com/gif.latex?X%3D%7Bs_%7B1%7D%2Cs_%7B2%7D%2C...%2Cs_%7BN%7D%7D), purpose dimension d
  
  * Output : conversion matrix U, mean vector ![equation](https://latex.codecogs.com/gif.latex?%5Cbar%7Bs%7D)
  
  * Algorithm
  
    1. Calculate X 's mean vector  ![equation](https://latex.codecogs.com/gif.latex?%5Cbar%7Bs%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7Ds_%7Bi%7D)
    
    2. Extract mean vector from the data ![equation](https://latex.codecogs.com/gif.latex?for%20%28i%3D1%5Chspace%7B5%7D%20to%20%5Chspace%7B5%7D%20N%29%20%5Chspace%7B10%7D%20s_%7Bi%7D%27%20%3D%20s_%7Bi%7D-%5Cbar%7Bs_%7Bi%7D%7D)
    
    3. Calculate covariance matrix ![equation](https://latex.codecogs.com/gif.latex?%5Csum%20%5Chspace%7B6%7D%20from%20%5Chspace%7B6%7D%20s_%7Bi%7D%27%2C%201%5Cleq%20i%5Cleq%20N)
    
    4. Calculate eigenvector and eigenvalue from ![equation](https://latex.codecogs.com/gif.latex?%5Csum)
    
    5. Select biggest d number of eigenvalue. And let's say ![equation](https://latex.codecogs.com/gif.latex?u_%7B1%7D%2C%20u_%7B2%7D%2C%20...%20%2C%20u_%7Bd%7D)
    
    6. Make conversion matrix U with result of 5.
    
    7. return ![equation](https://latex.codecogs.com/gif.latex?U%2C%20%5Chspace%7B4%7D%20%5Cbar%7Bs%7D)
    

### Algorithm to Extract Feature using PCA
  * Input : Conversion matrix U, mean vector ![equation](https://latex.codecogs.com/gif.latex?%5Cbar%7Bs%7D), sample s
  
  * Output : Feature vector x
  
  * Algorithm :
  
    ![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%201.%20%5Chspace%7B5%7D%20s%3Ds-%5Cbar%7Bs%7D%20%5Cnewline%202.%20%5Chspace%7B5%7D%20x%20%3D%20Us%20%5Cnewline%203.%20%5Chspace%7B5%7D%20return%20%5Chspace%7B5%7D%20x)
     



### Limitation
1. What if the principal component should be curved plane?

  ; Need to be flattened or should find out submanifold using kernel

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/pca_limit.PNG" width="500" height="200">
    

2. Is the direction of biggest variance is the most important?

  ; Not always true.
 
3. Should principal components be orthogonal to each other?

  ; In real case, data could be observed without careful criteria.
  



## LDA(Linear)

## Definition
If **PCA** was dimensional reduction in point of optimal *representation* of data,
**LDA** focuses on dimensional reduction in point of optimal *classification* of data.

So this aims to minimize within-class scatter and maximize between-class scatter.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/LDA.PNG" width="500" height="300">
    

# 0802-kernel_PCA.md
# 02/08/18 Daily Report


## Kernel-PCA

### PCA vs Kernal PCA

The standard PCA always finds linear principal components to represent the data in lower dimension. Sometime, we need non-linear principal components.If we apply standard PCA for the below data, 
it will fail to find good representative direction. Kernel PCA (KPCA) rectifies this limitation.
  
  - Kernel PCA just performs PCA in a new space.
  - It uses Kernel trick to find principal components in different space (Possibly High Dimensional Space).
  - PCA finds new directions based on covariance matrix of original variables. It can extract maximum P (number of features) eigen values. KPCA finds new directions based on kernel matrix. It can extract n (number of observations) eigenvalues.
  - PCA allow us to reconstruct pre-image using few eigenvectors from total P eigenvectors. It may not be possible in KPCA.
  - The computational complexity for KPCA to extract principal components take more time compared to Standard PCA.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/pca_kpca.png" width="500" height="300">

### Reference

[quora PCA vs kernel PCA](https://www.quora.com/Whats-difference-between-pca-and-kernel-pca)

# 0803-compare_autoencoder_pca.md
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

# 0803-simple_autoencoder.md
# 03/08/18 Daily Report


## Simple Autoencoder Code

The code shows simple implementation of autoencoder experiment on MNIST.


```python
import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')

```

### Reference

[code from L1aoXingyu](https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py)


# 0804-stacked_denoising_sparse.md
# 04/08/18 Daily Report

## Stacked Autoencoder (SAE)

As we can extract various features using many hidden layers, autoencoder can be implemented using many hidden layers.

This structure is called **Stacked Autoencoder**.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/stacked_autoencoder.png" width="500" height="500">


On above picture, we can figure out that *feature 1* is computed by given input, and *feature 2* is computed by *feature 1*.

We can figure out that each hidden layer shows more compact representation compared with prior layer.

Basically, this is stacked structure of autoencoder, and training is proceeded by **"Greedy Layer-Wise Training"**.

### Greedy Layer-Wise Training

Before 2006, researchers had difficulty on training network which has more than two hidden-layers.

As 2006 has come, three famous researchers found out how to train deep layers of network. The following papers provided solution.

  * Hinton - A Fast Learning Algorithm for Deep Belief Nets
  
  * Bengio - Greedy Layer - Wise Training for Deep Networks
  
  * LeCun - Efficient Learning of Sparse Representations with an Energy-Based Model
  
By these people's research, difficulty of training deep layered network had been solved. The problem was *vanishing gradient* and *overfitting*.



One of solution is  **Greedy Layer-Wise Training**. The concept is shown on the below picture.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/greedy_layer_wise_training.PNG" width="500" height="400">


As the term says, training is performed by layer in greedy way. Each layer is trained, assuming there is no layer beyond the present layer.

In addition, prior layer's parameter is fixed and present layer's parameter is updated.


#### Unsupervised Pre-training

The **Greedy Layer-Wise Training** can be used not only for Stacked AE, but also for network of supervised learning or CNN.

When training data exists enough, but labeled data is not abundant, pre-training can be done using unlabeled training data. This is called
**unsupervised pre-training**.

This kind of training way has been used since 2006. However after 2010, as activation function *RELU* appeared and *dropout*,
*maxout*, *data augmentation* and *batch-normalization* has been published, this way is rarely used now. Because just using supervised learning, it has been possible to get good performance.







## Denoising Autoencoder(DAE)

Denoising Autoencoder was published by Pascal Vincent and Yoshua Bengio, from paper "Extracting and Composing Robust Features with Denoising Autoencoder".

Before that, Autoencoder had been used to extract important feature from the given input data utilizing supervised learning and also for pre-training.


Then why the **"Denoising" Autoencoder** appeared?

### Concept 

 * Main Idea
 
  Even though there is noise in the input data, if the important feature is maintained, then output can show good enough reconstructed data. 
  
  <img src="https://github.com/jwcse/DeepLearning/blob/master/img/DAE_concept.PNG" width="700" height="300">


 * Process
  1. Add denoise to original input. 
  
  2. Put noised input to Autoencoder.
  
  3. Train the noised input to resemble original input.

  	
### Effect of Denoise Ratio

 <img src="https://github.com/jwcse/DeepLearning/blob/master/img/DAE_denoise_ratio.PNG" width="400" height="800">

 
[referenced from here](https://laonple.blog.me/220891144201)


### Code

Referenced from [here](https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/blob/master/07_Denoising_Autoencoder/Denoising_Autoencoder.py)

```python
# Simple Convolutional Autoencoder
# Code by GunhoChoi

import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Set Hyperparameters

epoch = 100
batch_size = 100
learning_rate = 0.0002

# Download Data

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)

# Encoder 
# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                 stride=1, padding=0, dilation=1,
#                 groups=1, bias=True)
# batch x 1 x 28 x 28 -> batch x 512

# Encoder 
# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                 stride=1, padding=0, dilation=1,
#                 groups=1, bias=True)
# batch x 1 x 28 x 28 -> batch x 512

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,32,3,padding=1),   # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,32,3,padding=1),   # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),  # batch x 64 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,64,3,padding=1),  # batch x 64 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)   # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),  # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.Conv2d(128,128,3,padding=1),  # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,256,3,padding=1),  # batch x 256 x 7 x 7
                        nn.ReLU()
        )
        
                
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out
    
encoder = Encoder().cuda()

# Decoder 
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
#                          stride=1, padding=0, output_padding=0,
#                          groups=1, bias=True)
# output_height = (height-1)*stride + kernel_size - 2*padding + output_padding
# batch x 512 -> batch x 1 x 28 x 28

# Decoder 
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
#                          stride=1, padding=0, output_padding=0,
#                          groups=1, bias=True)
# output_height = (height-1)*stride + kernel_size - 2*padding + output_padding
# batch x 512 -> batch x 1 x 28 x 28

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1), # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,128,3,1,1),   # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,64,3,1,1),    # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.ConvTranspose2d(64,64,3,1,1),     # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64,32,3,1,1),     # batch x 32 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,32,3,1,1),     # batch x 32 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,1,3,2,1,1),    # batch x 1 x 28 x 28
                        nn.ReLU()
        )
        
    def forward(self,x):
        out = x.view(batch_size,256,7,7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

decoder = Decoder().cuda()

# Noise 

noise = torch.rand(batch_size,1,28,28)

# loss func and optimizer
# we compute reconstruction after decoder so use Mean Squared Error
# In order to use multi parameters with one optimizer,
# concat parameters after changing into list

parameters = list(encoder.parameters())+ list(decoder.parameters())
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# train encoder and decoder

try:jupyter 
	encoder, decoder = torch.load('./model/deno_autoencoder.pkl')
	print("\n--------model restored--------\n")
except:
	pass

for i in range(epoch):
    for image,label in train_loader:
        image_n = torch.mul(image+0.25, 0.1 * noise)
        image = Variable(image).cuda()
        image_n = Variable(image_n).cuda()
        optimizer.zero_grad()
        output = encoder(image_n)
        output = decoder(output)
        loss = loss_func(output,image)
        loss.backward()
        optimizer.step()
        break
        
    torch.save([encoder,decoder],'./model/deno_autoencoder.pkl')
    print(loss)

# check image with noise and denoised image\

img = image[0].cpu()
input_img = image_n[0].cpu()
output_img = output[0].cpu()

origin = img.data.numpy()
inp = input_img.data.numpy()
out = output_img.data.numpy()

plt.imshow(origin[0],cmap='gray')
plt.show()

plt.imshow(inp[0],cmap='gray')
plt.show()

plt.imshow(out[0],cmap="gray")
plt.show()

print(label[0])

```

#### Result

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/DAE_ex_result.PNG" width="500" height="300">
	


### Reference
[This video](https://www.youtube.com/watch?v=t2NQ_c5BFOc&feature=youtu.be) explains on DAE, of which speaker is co-author of the paper, Hugo Larochelle. 






## Sparse Autoencoder


### Complete / Overcomplete

Usually siganl can be represented using linear combination of basis functions like case of Fourier or Wavelet.

In most case, dimension of basis function is equal to dimension of the input data. In this case, we say "**complete**".

![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20x%20%3D%20As%20%5Cnewline%20x%20%5Cin%20R%5E%7Bn%7D%2C%20A%20%5Cin%20R%5E%7Bn%5Ctimes%20n%7D%5Cnewline%20s%20%5Cin%20R%5E%7Bn%7D)

The above equation  shows that data x is derived from set of basis function A multiplies vector s. And dimension is all equal to n.

In this case, if A is determined, vector s becomes the representation of vector x. 

If it's complete, dimension of x and x is same and s should be unique.



Now let's look at another case.

![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20x%20%3D%20As%20%5Cnewline%20x%20%5Cin%20R%5E%7Bn%7D%2C%20A%20%5Cin%20R%5E%7Bn%5Ctimes%20m%7D%5Cnewline%20s%20%5Cin%20R%5E%7Bm%7D%20%5Chspace%7B7%7D%28m%20%3E%20n%29)

This shows **sparse coding**, that basis function's dimension is bigger than original data's dimension (m > n).

We say **overcomplete**, when representation vector's dimension(s's dimension) is bigger than the original input data's.

The advantage of having an over-complete basis is that our basis vectors are better able to capture structures and patterns inherent in the input data. 

When it comes to complete case, it's unique to represent the data, but in case of overcomplete, we need to select one with specific criteria because it's not unique.

Therefore, in sparse coding, we introduce the additional criterion of sparsity to resolve the degeneracy introduced by over-completeness.

Here, we define sparsity as having few non-zero components or having few components not close to zero.


### Sparse Coding

Sparse coding is one way of supervised learning and it was developed to represent data more efficiently based on overcomplete basis vector.

The picture below shows concept of sparse coding.


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/sparse_coding.PNG" width="700" height="500">
	

In the picture, Dictionary D contains basis vectors and data X is represented by ![equation](https://latex.codecogs.com/gif.latex?%5Calpha).

In this case, only three elements are not zero, so we can say that much dense representation has been enabled.




But important thing here is, how can we get D and ![equation](https://latex.codecogs.com/gif.latex?%5Calpha).

Let's look at sparse coding cost function

![equation](https://latex.codecogs.com/gif.latex?arg%20%5Chspace%7B1%7D%20%5Cmin_%7BD%2C%20A%7D%5Cleft%20%5C%7C%20X-AD%20%5Cright%20%5C%7C_%7BF%7D%5E%7B2%7D%20&plus;%20%5Cbeta%5Csum_%7Bi%2Cj%7D%5Cleft%20%7C%20a_%7Bi%2Cj%7D%20%5Cright%20%7C)

The 1st term is a reconstruction term which tries to force the algorithm to provide a good representation of given X using ![equation](https://latex.codecogs.com/gif.latex?AD).

And the 2nd term can be interpreted as a sparsity panalty which forces our representation of x to be sparse.

(cf. Sparsity penalty function can be log penalty too. In that case, gradient-based methods can be used)

The constant ![equation](https://latex.codecogs.com/gif.latex?%5Cbeta) is a scaling constant to determine the relative importance of these two contributions.




### What is Sparse Autoencoder


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/sparse_autoencoder.PNG" width="800" height="600">

Basically, when you train an autoencoder, the hidden units in the middle layer would fire (activate) too frequently, for most training samples. We don’t want this characteristic. We want to lower their activation rate so that they only activate for a small fraction of the training examples. This constraint is also called the *sparsity constraint*. It is sparse because each unit only activates to a certain type of inputs, not all of them.



#### Problem in Sparse Coding & k-Sparse Autoencoder

Sparse coding basically consists of two steps ; *dictionary learning* and *sparse encoding*.

In *dictionary learning*, dictionary and sparse code vector is derived by training data. In this case, sparsity condition should be satisfied, but it's usually not convex function. So, whenver we use gradient-based method, there can be risk of being local minimum.

To handle with this problem, a lot of methods had been published, but performance and result had been not good.


In this situation, **k-Sparse Autoencoder** provided efficient way for sparse coding.

By making constraints on activation of hidden layer at most k, sparsity condition is applied.

Until k-th neuron, the result is utilized and the rest is set zero.

In backpropagation, neuron with non-zero activation value is updated.

These process can be regarded as *dictionary learning*, as dictionary's atom is derived with iterative training.

To restate, only activating k-neurons and training weight matrix(dictionary) is identical to *dictionary learning* of **sparse coding**.

The table below shows the procedure.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/sparse_autoencoder_process.PNG" width="550" height="400">


In the table, ![equation](https://latex.codecogs.com/gif.latex?supp_%7Bk%7D%28W%5E%7BT%7Dx&plus;b%29) is equal to deriving k-support vector, and process 3) is equal to updating atom with that vector.


In encoding process, the number is ![equation](https://latex.codecogs.com/gif.latex?%5Calpha%20k) not k, because it is more efficient to use ![equation](https://latex.codecogs.com/gif.latex?%5Calpha) which is little bit bigger than 1, so that activating little more than k neurons. 

Referenced from [Laon People](https://laonple.blog.me/220943887634)


### Efficiency of k-Sparse Autencoder


The performance of **sparse coding** is basically determined by performance of dictionary. So, in order to get good performance, it is significant to derive good weight matrix.

Here, the good performance can be restated as
	
	-> Low similarity between atom included in a dictionary
		
	-> Representation of an atom using several atoms with linear combination should be difficult
	
	-> Coherence between atom should be small 
	
	-> Inner product between atom should be small!
	

Therefore, determining value k should be considerate to satisfying this :

![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20%5Cmu%28W%29%20%3D%20max_%7Bi%2Cj%7D%5Cleft%20%7C%20%3Cw_%7Bi%7D%2Cw_%7Bj%7D%3E%20%5Cright%20%7C%5Cnewline%20%5Cnewline%20k%20%5Cleq%20%281&plus;%5Cmu%5E%7B-1%7D%29)


### Performance by k-value

If k value is big, local feature can be extracted. In this case, it is not appropriate for classification, but it's useful for pre-training.

If k value is small(when sparsity is emphasized), too global feature can be extracted, so similar object could be classified as different category.

The picture below shows visualized filters depending on different k values.

We can see that (a) shows too local features and (d) shows too global feature.

The best result is hown on (c).

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/sparse_autoencoder_result.PNG" width="600" height="500">


In the below table, we can figure out that appropriate k-value of k-Sparse Autoencoder shows good performance compared with other unsupervised learning methods.


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/sparse_autoencoder_result2.PNG" width="500" height="500">



# 0806-CAE.md
# 06/08/18 Daily Report

## Convolutional Autoencoder(CAE)

### Intro

The concept of Convolutional Autoencoder was first published on 2011 by Jonathan Masci from Swiss, with name of 
[Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction](http://pdfs.semanticscholar.org/1c6d/990c80e60aa0b0059415444cdf94b3574f0f.pdf)

After that good papers are being published. 

Here is one interesting paper, [DeepPainter: Painter Classification Using Deep Convolutional Autoencoders](http://elidavid.com/pubs/deeppainter.pdf) which was published at 2016.



Usually, a painter's artifact is not that much and it is difficult to utilize *data augmentation* to increase training data. 

Therefore, training with small quantity of training data in supervised learning way can lead to overfitting problem.

So we can consider unsupervised learning like using Autoencoder.



### DeepPainter

The DeepPainter trains with unsupervised learning. In order to do so, fully connected layer is seperated and convolutional layer and pooling layer part is trained with Autoencoder-way.

After training, fully connected layer part is fine-tuned in supervised learning-way.

So, front part is form of Stacked Convolutional Autoencoder and Decoder part should be added in order to train.


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/CAE_overview.PNG" width="750" height="200">

In DeepPainter, max-pooling is used but this can be problem when constitute decode network, because the extracted maximum value in the pooling window can't be placed in appropriate position when size is shrinked.

So, in DeepPainter, pooling position is saved, so that we can figure out appropriate position when unpooling.

The picture below shows the way.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/CAE_unpooling.PNG" width="500" height="400">



After this process, training is proceeded as traditional training way of Autoencoder, which means that specific labeling is not needed because it is proceeded in unsupervised learning-way.

In unsupervised learning, training data was noised as case of Denoising Autoencoder.

After finish of training, decoder is eliminated and fully connected layer for classification is connected.


Now, classifier part is trained. Because front part is already trained, this can be categorized as fine-tuning.




#### Performance Result of DeepPainter

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/CAE_performance.PNG" width="600" height="500">


### Example Code

Referenced from [GunhoChoi's code](https://github.com/GunhoChoi/PyTorch-FastCampus/blob/master/08_Autoencoder/1_Convolutional_Autoencoder.ipynb)

```python

"""
    Referenced from 
    https://github.com/GunhoChoi/PyTorch-FastCampus/blob/master/08_Autoencoder/1_Convolutional_Autoencoder.ipynb
   
    Convolutional Autoencoder
        - MNIST dataset
        - Convolutional Neural Network
        - 2 hidden-layers
"""


## Settings
# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Set hyper-parameters
batch_size  = 100
learning_rate = 0.0002
num_epoch = 1


## Data
# Download Data
mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Set DataLoader
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, 
                    shuffle=True,num_workers=2,drop_last=True)

test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, 
                    shuffle=False,num_workers=2,drop_last=True)


## Model & Optimizer
# Model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # batch x 16 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),  # batch x 64 x 7 x 7
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out


encoder = Encoder().cuda()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = x.view(batch_size, 256, 7, 7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


decoder = Decoder().cuda()


# Loss func & Optimizer
parameters = list(encoder.parameters())+ list(decoder.parameters())
loss_func = nn.MSELoss()
optimizer = optim.Adam(parameters, lr=learning_rate)



## Train

try:
    encoder, decoder = torch.load('./model/conv_autoencoder.pkl')
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        optimizer.zero_grad()

        image = Variable(image).cuda()
        output = encoder(image)
        output = decoder(output)
        loss = loss_func(output, image)

        loss.backward()
        optimizer.step()

    if j % 10 == 0:
        torch.save([encoder, decoder], './model/conv_autoencoder.pkl')
        print(loss)


## Check with Train image
out_img = torch.squeeze(output.cpu().data)
print(out_img.size())

for i in range(5):
    #plt.imshow(torch.squeeze(image[i]).numpy(),cmap='gray')
    #plt.show()
    plt.imshow(out_img[i].numpy(),cmap='gray')
    plt.show()




```



