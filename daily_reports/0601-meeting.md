
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



