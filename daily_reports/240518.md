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
