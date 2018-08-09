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
