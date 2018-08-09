# 05/07/18 Daily Report


## Independence and Graphical Model

### Independence
  Two events are called **independent** if and only if P(A∩B)=P(A)P(B) (or equivalently, P(A∣B)=P(A)). And this can be denoted as A ⊥ B.
  The independence is equivalent to saying that observing B does not have any effect on the probability of A.
  

### Graphs & Independent Sets
A graph 𝐺 = (𝑉, 𝐸) is defined by a set of vertices 𝑉 and a set of edges 𝐸 ⊆ 𝑉 × 𝑉 (i.e., edges correspond to pairs of vertices).

A set 𝑆 ⊆ 𝑉 is an independent set if there does not exist an edge in 𝐸 joining any pair of vertices in 𝑆.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/independent_set_ex.PNG" width="600" height="900">


### Representation of Independence Structure with Graphical Model 
The amount of storage and the complexity of statistical inference are both affected by the independence structure of the joint probability distribution
  - **More independence means easier computation and less storage**
  - Want models that somehow make the underlying independence assumptions explicit, so we can take advantage of them
  (expensive to check all of the possible independencerelationships)
  - The simple, still powerful model!



## DGM(Directed Graphical Model) a.k.a Bayesian Networks
  
### Definition
A family of probability distributions that admit a compact parametrization that can be naturally described using a directed graph.
A Bayesian network is a directed graphical model that represents independence relationships of a given probability distribution.

* Directed acyclic graph (DAG), 𝐺 = (𝑉, 𝐸)
    - Edges are still pairs of vertices, but the edges (1,2) and (2,1) are now distinct in this model
* One node for each random variable
* One conditional probability distribution per node
* Directed edge represents a direct statistical dependence
* Corresponds to a factorization of the joint distribution

    ![equation](https://latex.codecogs.com/gif.latex?p%28x_%7B1%7D%2C...%2Cx_%7Bn%7D%29%20%3D%20%5Cprod_%7Bi%7Dp%28x_%7Bi%7D%7Cx_%7Bparents%28i%29%7D%29)

### Example
As an example, consider a model of a student’s grade g on an exam; this grade depends on several factors: 
the exam’s difficulty d, the student’s intelligence i, his SAT score s; it also affects the quality l of the reference letter from the professor who taught the course. 

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/bays_net_ex.png" width="500" height="300">

Each variable is binary, except for g, which takes 3 possible values.
Bayes net model describing the performance of a student on an exam. The distribution can be represented a product of conditional probability distributions specified by tables. 
The form of these distributions is described by edges in the graph. The joint probability distribution over the 5 variables naturally factorizes as follows

  ![equation](https://latex.codecogs.com/gif.latex?p%28l%2Cg%2Ci%2Cd%2Cs%29%3Dp%28l%7Cg%29p%28g%7Ci%2Cd%29p%28i%29p%28d%29p%28s%7Ci%29)

The graph clearly indicates that the letter depends on the grade, which in turn depends on the student’s intelligence and the difficulty of the exam.


### Bayesian networks are generative models
In the above example, to determine the quality of the reference letter, we may first sample an intelligence level and an exam difficulty; 
then, a student’s grade is sampled given these parameters; finally, the recommendation letter is *generated* based on that grade.


### Independencies described by Directed Graphs
Independencies can be recovered from the graph by looking at three types of structures.
 <img src="https://github.com/jwcse/DeepLearning/blob/master/img/three_types_components_of_bayes.PNG" width="400" height="300">
  * Common parent
  
    If G is of the form A←B→C, and B is observed, then A⊥C∣B. 
    
    However, if B is unobserved, then A⊥̸C. Intuitively this stems from the fact that B contains all the information that determines the outcomes of A and C
  * Cascade
    
    If G equals A→B→C, and B is again observed, then, again ​A⊥C∣B. However, if B is unobserved, then A⊥̸C. 
    Here, the intuition is again that B holds all the information that determines the outcome of C;
    thus, it does not matter what value A takes
  
  * V-structure
  
    If G is A→C←B, then knowing C couples A and B. In other words, A⊥B if C is unobserved, but A⊥̸B∣C if C is observed.
    
These structures clearly describe the independencies encoded by a three-variable Bayesian net. We can extend them to general networks by applying them recursively over any larger graph. This leads to a notion called **d-separation** (where d stands for directed).

We say that Q, W are d-separated when variables O are observed if they are not connected by an active path. An undirected path in the Bayesian Network structure G is called *active* given observed variables O if for every consecutive triple of variables X,Y,Z on the path, one of the following holds:
  - X←Y←Z, and Y is unobserved Y∉O
  - X→Y→Z, and Y is unobserved Y∉O
  - X←Y→Z, and Y is unobserved Y∉O
  - X→Y←Z, and Y or any of its descendents are observed.
  
 The notion of d-separation is useful, because it lets us describe a large fraction of the dependencies that hold in our model.

#### Reference :
[Stanford CS 228: Probabilistic Graphical Models - Bays Net](https://ermongroup.github.io/cs228-notes/representation/directed/)

[UTD slides](http://www.utdallas.edu/~nrr150130/cs6347/2016sp/lects/Lecture_2_Bayes.pdf)
