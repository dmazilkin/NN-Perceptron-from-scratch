# Perceptron binary classification realization from scratch
This repo contains Perceptron realization from scratch for solving *binary classification problem*. There are two main files:
1. **perceptron_sgd.py** - contains Perceptron realization with *Stochastic Gradient Descent* optimizer and *sign* activation function,
2. **main.ipynb** - contains visualization of how Perceptron works on generated dataset. 
# Usage 
The Perceptron usage is pretty simple - just import from *perceptron.py* class **Perceptron** and initialize it with X matrix, Y vector, custom learning rate (default 0.01 is set) and custom maximum number of epoch (default 3000 is set).
Then simply call *.train()* method and Perceptron training process will begin. After training is completed, you can do predictions with *.predict()* methods. 

An example of use can be seen in *main.ipynb*.