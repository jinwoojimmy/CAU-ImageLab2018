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




