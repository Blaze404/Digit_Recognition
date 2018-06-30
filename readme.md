Digit Classification in Python

Features:
   Digit classification using neural networks with two hidden layers and tanh, tanh and softmax as activation functions
   Made with numpy as only dependency in python 3.
   The structure of neural network
   784 neurone in first layer
   30 neurons in hidden layer 1
   30 neurons in hidden layer 2
   and 10 neurons in output layer signifying each digit

Use:
    just run main.py, with appropriate function, and commenting out other call to functions

    => initial_training = this was used to train the model for the first time
    => updations( train=True/False ): if train = True it will further train the network,
                    and if train = False, it will just show the performans on train and test data
    => check_performance => check performance on custom dataset
    => check_on_custom_image = give image path as input and check model on that image
                        Important: Image should be square shapd ( z*z pixel in size )