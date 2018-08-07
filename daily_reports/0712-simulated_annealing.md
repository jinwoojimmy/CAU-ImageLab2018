# 12/07/18 Daily Report


## Simulated Annealing

### What is Simulated Annealing
The simulated annealing algorithm was originally inspired from the process of annealing in metal work. 

Annealing involves heating and cooling a material to alter its physical properties due to the changes in its internal structure. As the metal cools its new structure becomes fixed,
consequently causing the metal to retain its newly obtained properties. 
In simulated annealing we keep a temperature variable to simulate this heating process. 
We initially set it high and then allow it to slowly 'cool' as the algorithm runs. 
While this temperature variable is high the algorithm will be allowed, with more frequency, to accept solutions that are worse than our current solution. This gives the algorithm the ability to jump out of any local optimums it finds itself in early on in execution. As the temperature is reduced so is the chance of accepting worse solutions, therefore allowing the algorithm to gradually focus in on a area of the search space in which hopefully, a close to optimum solution can be found. This gradual 'cooling' process is what makes the simulated annealing algorithm remarkably effective at finding a close to optimum solution when dealing with large problems which contain numerous local optimums.
The nature of the traveling salesman problem makes it a perfect example.

### Advantages 

You may be wondering if there is any real advantage to implementing simulated annealing over something like a simple hill climber. Although hill climbers can be surprisingly effective at finding a good solution, they also have a tendency to get stuck in local optimums. As we previously determined, the simulated annealing algorithm is excellent at avoiding this problem and is much better on average at finding an approximate global optimum.

To help better understand let's quickly take a look at why a basic hill climbing algorithm is so prone to getting caught in local optimums.

A hill climber algorithm will simply accept neighbour solutions that are better than the current solution. When the hill climber can't find any better neighbours, it stops.

To compare with gradient descent, gradient descent works only if we have a good initial segmentation. However, simulated annealing always works(at least in theory).



### Algorithm


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/sim_anneal_alg_structure.jpg" width="420" height="600">
    
### Reference

[simulated-annealing-algorithm-for-beginners](http://www.theprojectspot.com/tutorial-post/simulated-annealing-algorithm-for-beginners/6)
    
