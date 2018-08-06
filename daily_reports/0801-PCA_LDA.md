# 01/08/18 Daily Report


## PCA(Principal Component Analysis)

### Basic Knowledge

* **Variance** : 
  Represents "how much the data is far from the average"

* **Covariance** : 
  Represents "relationship between group of data points"
  
### Definition
A way of linear dimensional reduction which aims to figure out the axis representing the dataset best.

The *axis* is equal to 

  = Spreading the data most broadly
  
  = Having the biggest variance

  = **Principal component**
  
  
### Assumption of PCA
1. Submanifold representing data has *linear basis*.

2. The vector of biggest variance contains the most information.

3. Principal components are orthogonal each other.
  
### Limitation
1. What if the principal component should be curved plane?

  ; Need to be flattened or should find out submanifold using kernel

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/pca_limit.PNG" width="500" height="200">
    

2. Is the direction of biggest variance is the most important?

  ; Not always true.
 
3. Should principal components be orthogonal to each other?

  ; In real case, data could be observed without careful criteria.
  



## LDA(Linear)

## Definition
If **PCA** was dimensional reduction in point of optimal *representation* of data,
**LDA** focuses on dimensional reduction in point of optimal *classification* of data.

So this aims to minimize within-class scatter and maximize between-class scatter.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/LDA.PNG" width="500" height="300">
    
