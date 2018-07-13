# 27/06/18 Daily Report


## Various kinds of Activation Functions

There are a lot of functions for activation function in neural net. 

Each activation function has own characteristic, and we should choose activation function by 

### 1. Activation Function
- Function that convert weighted sum into output signal in a neuron. The output contains strength of signal, not binary value.

- Provides non-linearity to the neural net and this enables effect of stacked layers.

- Monotonic(not necessarily), so error surface assosiated with single-layer model is guaranteed to be convex.

- Continuously differentiable ; this is desirable property. In backpropagation among training, gradient-based optimization can be applied.


### 2. Sigmoid Function

- Definition
  
  ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-x%7D%7D)
    
    
- Characteristic
  - Zero-centered, converges to zero as x decreases. Counterwise, converges to one as x increases.
  - *Vanishing Gradient Problem* :
    If output value by activation function is really close to zero or one, derivative will be close to zero.
    In backpropagation process of training, weight would not be updated if the derivative value is almost zero, because update is proceeded by chain rule.
  - Activation value(output) is always over zero
  - Computation of e^x  -> expensive 
  
  
  ### 3. Tanh (Hyperbolic Tangent)
  
  - Definition
    
    ![equation](https://latex.codecogs.com/gif.latex?tanh%28x%29%20%3D%202*sigmoid%282x%29%20-%201%20%3D%20%5Cfrac%7Be%5E%7Bx%7D-e%5E%7B-x%7D%7D%7Be%5E%7Bx%7D&plus;e%5E%7B-x%7D%7D)

  - Characteristic
    - Output range : **[**-1, 1 **]**
    - Trained faster than sigmoid.
    - Still vanishing gradient problem exists.
    
### 4. Relu (Rectified Linear Unit)
  
  - Definition
    
    ![equation](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20max%280%2Cx%29)
    
  - Characteristic
    - Don't need to worry about vanishing gradient problem ; if x is positive, gradient value is equal to one.
    - Converges faster ; Easy to get derivative value and complexity of computation is low.
    - When x is negative, gradient becomes zero.(Cons)
    
### 5. Leaky Relu
 - Definition
    
    ![equation](https://latex.codecogs.com/gif.latex?f%28x%29%3Dmax%280.01x%2C%20x%29)
    
  - Characteristic
    - Cover cons of Relu.
    - Gradient is 0.01 when x is positive and 1 when positive.
    
    
### 6. ELU (Exponential Linear Units)
  
  - Definition
    
    ![equation](https://latex.codecogs.com/gif.latex?%5Cpar%7B%20%5Cnoindent%20%5Calpha%28e%5E%7Bx%7D-1%29%20%5Chspace%7B1cm%7D%20%28x%3C%3D0%29%20%5C%5C%20x%20%5Chspace%7B2.5cm%7D%20%28x%20%3E%200%29%20%7D)
    
  - Characteristic
    - Include positive aspects of Relu.
    - Computation of e^x  -> expensive 
    
### 7. Maxout
 - Definition
    
    ![equation](https://latex.codecogs.com/gif.latex?max%28w_%7B1%7D%5E%7BT%7Dx&plus;b_%7B1%7D%2C%20w_%7B2%7D%5E%7BT%7Dx&plus;b_%7B2%7D%29)
    
  - Characteristic
    - Generalized version of Relu and Leaky relu.
    - Compared with relu, has twice much of parameters -> expensive.
    
   
#### Graph     
![image](https://github.com/jwcse/DeepLearning/blob/master/img/activation_func_graph.png)
