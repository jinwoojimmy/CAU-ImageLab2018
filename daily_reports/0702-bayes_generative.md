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
