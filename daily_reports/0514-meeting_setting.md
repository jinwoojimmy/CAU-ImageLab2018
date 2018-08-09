
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
