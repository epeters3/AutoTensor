# CS 470 Lab 5 Report - Optimizing ML Model Hyperparameter Selection With Q-Learning

By Evan Peterson

## Introduction

From Dr. Crandall:

> "You can pick an AI-related project that has similar scope of work as the Reversi Cup 2018. Include some kind of write-up documenting what you did and documenting explorations into what makes the algorithm/system you created work and/or not work."

For this lab I chose to implement the Q-Learner algorithm. I am very interested in machine learning and deep learning. Deep learning models have many "hyper" parameters that one must set to the correct values in order to get optimal prediction accuracy. Optimal hyper parameter values change for each algorithm and dataset used. Also it takes a lot of problem domain expertise and deep learning expertise to know how to set them optimally. Because of this, I was interested in implementing a Q-Learner that could search a deep neural network's hyper parameter space, finding the shortest path to the optimal settings for a given dataset, to maximize predictive accuracy for that dataset.

## Implementation

My Q-Learner hyperparameter search module (which I precociously call "AutoTensor"), was written in Python and used the Keras library as a key 3rd party dependency. Keras does not provide any of the Q-Learner code, but it does give a high-level API to Google's very powerful deep learning library called tensorflow. I wrote my Q-Learner as a module that wraps Keras to search the hyper parameter space of a Keras deep learning network. There are a few key components which I created for my module:

### The `q_learning` Module

My `q_learning` module implements the Q Learning algorithm, and also manages the mapping of hyperparameter configurations for deep learning models to actions and states as fed into the Q Learning algorithm. Each "state" given to the Q Learner is a deep learning model configuration, and each action the Q Learner can take is a function that modifies a single hyperparameter of a model config, yeilding a new altered config, which to the Q Learner is the new state resulting from a state action pair.

### The `tensorflow` Module

My `tensorflow` module acts as a wrapping interface for the Keras library, exposing methods for building and running a deep learning network given a model configuration (a.k.a. a Q Learner's state), and for yielding the test set accuracy of that model as the reward the Q learner uses to compute the Q value of a given (state, action) pair.

### The `data_mgmt` Module

My `data_mgmt` module is responsible for loading and shaping datasets into separate features (the independent variables) and labels (the dependent variable to predict), which are also split into separate training, validation, and test sets. The module can load and prepare `.arff` data files as well as folders of images.

### How The Pieces Fit

My Q Learning implementation starts with an intial state, and a possible list of actions, and tries each action, computing the Q value for each (state, action) pair. Once an optimal action (in regards to Q-value) is determined for the current state, that action is officially taken and the resulting state is stored as the new current state. Q values are then again calculated for all possible actions taken from that new current state, and a best action is identified and taken, resulting again in a new current state. This process is repeated until no action can be taken from the current state that will improve rewards i.e. there is no action that will yield a higher Q value than that of the current state. In other words, the algorithm continues until it reaches a local maximum Q-value. My calculation of a Q value looks like this:

```python
new_q = curr_q + self.alpha * ((reward + self.discount * self.q_vals[new_state_i, best_action_i]) - curr_q)
```

Put to words, the `new_q` q value for a (`state`, `action`) pair is a weighted discounted sum of the reward for taking `action` from `state` plus the Q-value of taking the best possible action from the new state.

The traditional Q Learning algorithm iteratively calculates Q values for all (state, action) pairs until the q values converge. Due to the intense computational resources needed to train deep learning models, I chose to limit the search of the Q-value space in the way stated above for now. This allows the Q Learning to be applied in a way that can yeild a hyperparameter configuration for a deep learning network in a timely manner that also gives "optimal" predictive accuracy. I use quotations because It likely does not yield the (state, action) pair that provides the true global maximum for predictive accuracy, but it is a well informed local maximum, and does prove fairly effective as the "Experimentation" section shows.

## Experimentation

To help evaluate the effectiveness of my AutoTensor model, I chose to test it on two datasets. One is the iris dataset, which is a classic toy dataset (150 instances) used to classify species of Iris flower, given 4 measurements of each flower. The other datset was the MAGIC gamma-ray dataset; a 2-class classification problem which allows electromagnetic readings to be classified as either incoming gamma rays or background cosmic ray noise. For both datasets, I used a fully connected neural network with a single hidden layer and an output layer. The output layer uses the softmax activation algorithm to output a probability for each target class. It outputs the probability that the data instance belongs in each class.

Let's look at the Iris dataset first. Typical test set prediction accuracies for the Iris data are 85-95%. On the first training of the network, using a default model configuration state, an accuracy of 45.45% was reached. After trying each possible action, the Q Learner determind the next optimal step was to set the activation method of the fully connected layer to use the SeLu algorithm (instead of the Sigmoid method it was using before). That yielded a classification accuracy (used as the Q-Learner's reward) of 72.73%. Next, the Q-Learner identified setting the model's loss algorithm to binary crossentropy as the next optimal move (instead of the categorical crossentropy it was using before). It should be noted that the loss algorithm is used to determine how to calculate the errors for the network, which are backpropagated through to update the network weights, causing the model to learn. That step yielded classification accuracy of 87.88%. At this point, the Q Learner could not find an action that would increase the Q value further, given the set of 36 actions it had available to try, and ended execution.

Next I tried the Q Learner on the MAGIC dataset (6,666 instances). Typical test set prediction accuracies for the MAGIC data are 75-85%. Using the default configuration, the model gave an accuracy of 79.78%, which was very good. For the MAGIC data, the Q Learner identified a sequence of 3 actions to take that would increase predictive accuracy at each step. The first was setting the fully connected layer's activation method to the Softmax algorithm (it was using the Sigmoid method before). That yielded a prediction accuracy of 85.19%. Next, the Q Learner decided to double the number of nodes in the fully connected layer, to increase the number of possible feature combinations used to calculate a prediction score for each of the output classes. That yielded a prediction accuracy of 85.79%. Finally, the Q Learner decremented the patience of the model. Patience is how many training iterations (called epochs) the model will wait without seeing a gain in validation set accuracy before it decides to stop training on the data. The Q Learner decremented patience from 1 epoch to 0 epochs, giving a final predictive accuracy for the model of 86.19%.

## Conclusions & Limitations

This was a very fun project to implement, although it did take a while and the learning curve was steep. It was great to see it train on those two datasets and give somewhat competitive results. The main limitation of this model is the time it takes to get a reward for a given (state, action) pair, since that reward is determined by training a deep neural network. This does provide a nice alternative to exhaustively searching the full hyper parameter space of a model though, which in effect is an intractable problem. My current implementation of the Q Learner approach uses the Q Learning algorithm to conduct a type of greedy search through a more informed subset of the hyperparameter space. Given the computational resources e.g. access to CUDA enabled GPUs or the Fulton Supercomputing Lab on campus, I could perhaps open up the Q Learner to search the full (state, action) space like a traditional Q Learner and converge at optimal q values, and even more important to this problem, find a model configuration state that gives the true global maximum predictive accuracy for a given dataset.
