# 06/07/18 Daily Report

## UGM(Unirected Graphical Model) a.k.a MRF(Markov Random Fields)

### Intro - Necessity of UGM
Bayesian networks are a class of models that can compactly represent many interesting probability distributions.
However, some distributions cannot be perfectly represented by a Bayesian network.
  
In such cases, unless we want to introduce false independencies among the variables of our model, we must fall back to a less compact representation (which can be viewed as a graph with additional, unnecessary edges). 
This leads to extra, unnecessary parameters in the model, and makes it more difficult to learn these parameters and to make predictions.  

There exists, however, another technique for compactly representing and visualizing a probability distribution that is based on the language of undirected graphs. 
This class of models (known as Markov Random Fields or MRFs) can compactly represent distributions that directed models cannot represent.   
  
  
 ### Definition
 A Markov Random Field (MRF) is a probability distribution p over variables x1,...,xn defined by an undirected graph G in which nodes correspond to variables xi. 
 The probability p has the form
 
 ![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20p%28x_%7B1%7D%2C...%2Cx_%7Bn%7D%29%3D%5Cfrac%7B1%7D%7BZ%7D%5Cprod_%7Bc%5Cepsilon%20C%7D%20%5Cphi%20_%7Bc%7D%20%28x_%7Bc%7D%29%20%5Cnewline%5Cnewline%5Cnewline%20Z%20%3D%20%5Csum_%7Bx_%7B1%7D%2C...%2Cx_%7Bn%7D%7D%20%5Cprod_%7Bc%5Cepsilon%20C%7D%20%5Cphi%20_%7Bc%7D%20%28x_%7Bc%7D%29)
 
 where C denotes the set of cliques (i.e. fully connected subgraphs) of G. 
  * clique : clique (/ˈkliːk/ or /ˈklɪk/) is a subset of vertices of an undirected graph such that every two distinct vertices in the clique are adjacent; that is, its induced subgraph is complete. Cliques are one of the basic concepts of graph theory and are used in many other mathematical problems and constructions on graphs. 
 
 The value is a normalizing constant(a.k.a partition function) that ensures that the distribution sums to one.
 
 
### Example
As a motivating example, suppose that we are modeling voting preferences among persons A,B,C,D. Let’s say that (A,B), (B,C), (C,D), and (D,A) are friends, and friends tend to have similar voting preferences. 
These influences can be naturally represented by an undirected graph.

One way to define a probability over the joint voting decision of A,B,C,D is to assign scores to each assignment to these variables and then define a probability as a normalized score. A score can be any function, but in our case, we will define it to be of the form

![equation](https://latex.codecogs.com/gif.latex?%5Cnewline%20%5Ctilde%7Bp%7D%28A%2CB%2CC%2CD%29%20%3D%20%5Cphi%28A%2CB%29%5Cphi%28B%2CC%29%5Cphi%28C%2CD%29%5Cphi%28D%2CA%29%20%5Cnewline%20p%28A%2CB%2CC%2CD%29%3D%5Cfrac%7B1%7D%7BZ%7D%5Ctilde%7Bp%7D%28A%2CB%2CC%2CD%29)

And we can represent in graph like this

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/markvov_ex.PNG" width="600" height="400">
 
 
### Feature
Note that unlike in the directed case, we are not saying anything about how one variable is generated from another set of variables (as a conditional probability distribution would do). 
We simply indicate a level of coupling between dependent variables in the graph.
In a sense, this requires less prior knowledge, as we no longer have to specify a full generative story of how the vote of B is constructed from the vote of A (which we would need to do if we had a P(B∣A) factor). 
Instead, we simply identify dependent variables and define the strength of their interactions;
this in turn defines an energy landscape over the space of possible assignments and we convert this energy to a probability via the normalization constant.

  * Advantage
    - They can be applied to a wider range of problems in which there is no natural directionality associated with variable dependencies.

    - Undirected graphs can succinctly express certain dependencies that Bayesian nets cannot easily describe (although the converse is also true)
  
  * Disadvantage
    - Computing the normalization constant Z requires summing over a potentially exponential number of assignments. Can be NP-hard; thus many undirected models will be intractable and will require approximation techniques.
    
    - Undirected models may be difficult to interpret.
    
    - It is much easier to generate data from a Bayesian network, which is important in some applications.
  
It is not hard to see that Bayesian networks are a special case of MRFs with a very specific type of clique factor (one that corresponds to a conditional probability distribution and implies a directed acyclic structure in the graph), 
and a normalizing constant of one. In particular, if we take a directed graph G and add side edges to all parents of a given node (and removing their directionality), then the CPDs (seen as factors over a variable and its ancestors) factorize over the resulting undirected graph. 
The resulting process is called *moralization*.


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/moralization_ex.png" width="500" height="400">


The converse is also possible, but may be computationally intractable, and may produce a very large (e.g. fully connected) directed graph.
  
Thus, MRFs have more power than Bayesian networks, but are more difficult to deal with computationally. 
A general rule of thumb is to use Bayesian networks whenever possible, and only switch to MRFs if there is no natural way to model the problem with a directed graph (like in our voting example).  
 



#### Reference :
[Stanford CS 228: Probabilistic Graphical Models - MRF](https://ermongroup.github.io/cs228-notes/representation/undirected/)

[UTD slides](http://www.utdallas.edu/~nrr150130/cs6347/2016sp/lects/Lecture_4_MRFs.pdf)
