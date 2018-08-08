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

### Algorithm to get Convertion Matrix U
  * Input : training data ![equation](https://latex.codecogs.com/gif.latex?X%3D%7Bs_%7B1%7D%2Cs_%7B2%7D%2C...%2Cs_%7BN%7D%7D), purpose dimension d
  
  * Output : conversion matrix U, mean vector ![equation](https://latex.codecogs.com/gif.latex?%5Cbar%7Bs%7D)
  
  * Algorithm
  
    1. Calculate X 's mean vector  ![equation](https://latex.codecogs.com/gif.latex?%5Cbar%7Bs%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7Ds_%7Bi%7D)
    
    2. Extract mean vector from the data ![equation](https://latex.codecogs.com/gif.latex?for%20%28i%3D1%5Chspace%7B5%7D%20to%20%5Chspace%7B5%7D%20N%29%20%5Chspace%7B10%7D%20s_%7Bi%7D%27%20%3D%20s_%7Bi%7D-%5Cbar%7Bs_%7Bi%7D%7D)
    
    3. Calculate covariance matrix ![equation](https://latex.codecogs.com/gif.latex?%5Csum%20%5Chspace%7B6%7D%20from%20%5Chspace%7B6%7D%20s_%7Bi%7D%27%2C%201%5Cleq%20i%5Cleq%20N)
    
    4. Calculate eigenvector and eigenvalue from ![equation](https://latex.codecogs.com/gif.latex?%5Csum)
    
    5. Select biggest d number of eigenvalue. And let's say ![equation](https://latex.codecogs.com/gif.latex?u_%7B1%7D%2C%20u_%7B2%7D%2C%20...%20%2C%20u_%7Bd%7D)
    
    6. Make conversion matrix U with result of 5.
    
    7. return ![equation](https://latex.codecogs.com/gif.latex?U%2C%20%5Chspace%7B4%7D%20%5Cbar%7Bs%7D)
    

### Algorithm to Extract Feature using PCA
  * Input : Conversion matrix U, mean vector ![equation](https://latex.codecogs.com/gif.latex?%5Cbar%7Bs%7D), sample s
  
  * Output : Feature vector x
  
  * Algorithm :
  
    ![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%201.%20%5Chspace%7B5%7D%20s%3Ds-%5Cbar%7Bs%7D%20%5Cnewline%202.%20%5Chspace%7B5%7D%20x%20%3D%20Us%20%5Cnewline%203.%20%5Chspace%7B5%7D%20return%20%5Chspace%7B5%7D%20x)
     



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
    
