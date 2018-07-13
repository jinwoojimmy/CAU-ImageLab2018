# 29/06/18 Daily Report

## Basic Conception of Probability

### Marginal Probability
Probability of any single event *occurring unconditioned* on any other events.

Whenever someone asks you whether the weather is going to be rainy or sunny today, you are computing a marginal probability.

### Joint Probability 
Probability of *more than one event occurring simultaneously*. 

If I ask you whether the weather is going to be rainy and the temperature is going to be above a certain number, you are computing a **joint probability**.

### Conditional Probability 
Probability of an event occurring given some events that *you have already observed*. 

When I ask you what’s the probability that today is rainy or sunny given that I noticed the temperature is going to be above 80 degrees, you are computing a **conditional probability**.


-----------------


These three concepts are intricately related to each other. Any **marginal probability** can always be computed in terms of sums(**∑**) of **joint probabilities** by a process called **marginalization**.

Mathematically, this looks like this for two events A and B

##### P(A)=∑bP(A,b)

And **joint probabilities** can always be expressed as the product of **marginal probabilities** and **conditional probabilities** by the chain rule.

##### P(A,B)=P(A)⋅P(B|A)

These rules can be generalized to any number of events and are immensely useful in all type of statistical analysis.


##### Reference :
[quora answered by Nipun Ramakrishnan](https://www.quora.com/What-are-marginal-probability-and-conditional-probability)
