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


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/sparse_coding.PNG" width="400" height="300">
	

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


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/sparse_autoencoder.PNG" width="400" height="300">

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


