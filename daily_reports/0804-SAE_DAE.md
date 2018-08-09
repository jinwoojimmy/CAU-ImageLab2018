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




