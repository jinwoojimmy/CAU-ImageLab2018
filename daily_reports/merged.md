# 0514-meeting_setting.md

# 14/05/18 Daily Report

1. Meeting
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
  
2. Environment Configuration and Installation
- install anaconda3
- install pytorch
- install PyCharm IDE

3. Study on Pytorch Basics
- Pytorch includes various kinds of computational library and methods.

```{.python}
  
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
need such effort, and feature is extracted by relation between dataset. Autoencoder is one of the most famous example of
unsupervised learning, and this is what we are going to look at.

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




# 0521-autogradient_regression.md
# 21/05/18 Daily Report

## Auto Gradient

*torch.Tensor* is the central class of the package. If we set its attribute .requires_grad as True, it starts to track all operations on it. When we finish our computation we can call .backward() and have all the gradients computed automatically. The gradient for this tensor will be accumulated into .grad attribute.

If you want to compute the derivatives, you can call .backward() on a Tensor as mentioned above. If Tensor is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments to backward(), however if it has more elements, you need to specify a gradient argument that is a tensor of matching shape.


We've seen gradient descent operating by our handling. The code below is implementation of gradient descent using pytorch library.


[code](./codes/variableAutograd.py)

[another code](https://github.com/hunkim/PyTorchZeroToAll/blob/master/03_auto_gradient.py)
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


[code](https://github.com/jwcse/DeepLearning/blob/master/codes/logisticRegression.py)





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

# 0528-cnn.md
# 28/05/18 Daily Report


## CNN Architecture

*CNN(Convolutional Neural Network)* is neural network utilized in image processing , MLP and so on.
The key idea is *convolution*, which means only small portion of image is handled at a time. This small portion is called 'patch',
and whole input image is processed by size of the patch.

CNN is classifed as locally connected nerual net, and this kind of neural net produces smaller weights compared to fully connected neural net. 
And this enables flexible handling of input image data.

1. Convolutional Layer - creation of feature map

Within the size of patch, same size of filter(kernel) which contains weights are computed to input image. The computation is done by 
dot product. After computation, window is moved and computation is perfomed again. Each result of computation value is put into element of
feature map. The window is moved by step, called 'stride'. Also, 'padding' can be done to the original image, which means zero values can be added
to boundary of the input value array.
The output size should be different based on stride and patch size. For example, let's assume that input image size is 32x32x1,
filter size is 5x5x1, and stride is 1x1. Then, output array size would be 28x28.

In addition, depending on number of filters, output's depth would be different. Each filter's size is same but contains different values.
If we use 6 filters, for instance, the output's depth would be 6.

2. Pooling Layer - subsample

Purpose is to reduce information generated by convolutional layer.
One of the representative way is *max pooling*. With certain filter size and stride, maximum value in the area is put into the element of output.

3. Feedforward layer - classification ; fully connected layer

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
<img src="https://github.com/jwcse/DeepLearning/blob/master/img/generative-discriminative.jpg" width="400" height="300">

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

