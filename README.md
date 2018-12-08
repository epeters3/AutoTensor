# Using Q-Learning To Optimize Neural Net Hyperparameters

Author: Evan Peterson

## Credits

This project was inspired by these two bodies of work:

1. [The Google AutoML Project](https://ai.googleblog.com/2017/05/using-machine-learning-to-explore.html)
2. [Work done by Baker, Gupta, Naik, & Rasker](https://arxiv.org/abs/1611.02167)

## Brainstorming - DELETE ME

> "To find an appropriate model size, it's best to start with relatively few layers and parameters, then begin increasing the size of the layers or adding new layers until you see diminishing returns on the validation loss." - Tensorflow Documentation

**Possible states (TF Keras NN Hyperparameters)**:

-   In Layer Setup:
    -   Number of layers (`int`) (maybe don't try at first, adds complexity that I still need to reason through)
    -   Layer type
        -   Dropout rate (`double`)- "The fraction of the layer's features that are being zeroed-out; it is usually set between 0.2 and 0.5"
        -   The kernel regularizer
            -   Lx regularization coefficient (`double`)
-   In Compile Setup:
    -   Metrics
        -   `"categorical_accuracy"`
        -   `"mae"` - Mean absolute error
        -   `"accuracy"`
-   When training/fitting:

    -   Batch size (`int`)
    -   steps_per_epoch (`ing`) - "the number of training steps the model runs before it moves to the next epoch."

**Data to record**:

-   A history of all `(action, state)` pairs tried

**Things to Try**:

-   Epsilon-greedy exploration
-   Include `None` as an option on all valid hyperparameters
