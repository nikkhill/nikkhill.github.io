---
layout: post
title:  "Layered approach to coding Dense neural net"
date:   2018-11-23 10:28:25 -0500
categories: deeplearning ml
---

I was working through a homework for this [Machine Learning Course][ml-course]
and we were supposed to code a fully connected net from scratch using python and numpy. 
There was this really cool suggestion of doing that by using a layered approach. 
We keep using layers when we work with high level libraries like TensorFlow
and PyTorch but why not try code it from scratch. 
The motivation for this approach is when making it easy to add more layers if we want.
This is very important if we want to build complex networks. 

So you have a bunch of equations for your forward algorithm (which computes output given an input).
You are a calculus expert, so you derive all the backward equations
(which compute gradients of the loss with each parameter). 
Then you implement two python functions forward and backward using what these equations.
This was easy when you had one hidden layer or maybe two. 
But we want to be efficient about it. 
Let's create a `LinearLayer` and a `SigmoidLayer` which do the job they are named for.
One object of each of these layers will take care of just one hidden layer. 
You can combine them into something like DenseLayer, and you probably should, but let's keep it simple here. 


{% highlight python %}
class LinearLayer:
    # a = Wx + w0
    def __init__(self, input_dim, output_dim, weight_init_method):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = initialize_weights(weight_init_method, shape=(self.output_dim, self.input_dim))
        self.w0 = initialize_weights(weight_init_method, shape=(self.output_dim, 1))

    def forward(self, x):
        return self.w0 + np.matmul(self.w, x)

    def backward(self, x, ga):
        return np.matmul(ga, x.T), ga, np.matmul(self.w.T, ga)

    def update_weights(self, gw, gw0, learning_rate):
        self.w = self.w - learning_rate * gw
        self.w0 = self.w0 - learning_rate * gw0


class SigmoidLayer:
    # z = sigmoid(a)
    def forward(self, a):
        return np.reciprocal(1 + np.exp(-a)).reshape(a.shape[0], 1)

    def backward(self, z, gz):
        return (gz * (z * (1 - z))).reshape(gz.shape[0], 1)
{% endhighlight %}

TODO

[ml-course]: http://www.cs.cmu.edu/~mgormley/courses/10601bd-f18/
