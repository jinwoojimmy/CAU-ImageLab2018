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
