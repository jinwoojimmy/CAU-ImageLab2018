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
 

### Mathematical Representation 

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
 
 
 ### L2 vs L1
 
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
 
 ### Affine Transform
 With affine transform, we can get good enough training dataset.
 
 <img src="https://github.com/jwcse/DeepLearning/blob/master/img/data_aug_affine.PNG" width="700" height="500">
 
 ### Elastic Distortion
 
 Microsoft developed this method to generate effictive training dataset.
 
 As the picture below shows, generate displacement vector in diverse way.
 
 <img src="https://github.com/jwcse/DeepLearning/blob/master/img/elastic_distortion.PNG" width="600" height="400">
 
 
 ### Dropout
 
 As number of hidden layers increase, training performance gets better.
 
 But, when it comes to size of the layer, there's problem of overfitting.
 
 In such a case, dropout can be a good solution.
 
 #### Overview
 
 <img src="https://github.com/jwcse/DeepLearning/blob/master/img/elastic_dropout.PNG" width="600" height="400">
 
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
 
 
 
 
 
 
 
 
