import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf

# computes return for next step and uses it as input c for the exp_ut function
def returns_ut(a, r, d, D, w, s):
    c = r*w + d*s*D
    #return exp_ut(c,a)
    return exp_ut(c,a)

# takes risk aversion a and variable that the economic decision-maker prefers more of c
# return exponential utility U
def exp_ut(c, a):
    U = c
    if a != 0:
        U = 1 - np.exp(-c*a)/a
    return U

# Creating a Sequential Model and adding the layers
def init_model(N,n_layers,n_neurons):
    # initialize model so it can take an array of th cuurent cash and stocks for all agents
    model = Sequential()
    input = tf.keras.Input(shape=(2*N))
    # add as many layers as defined by n_layer, each with n_neurons
    for i in np.arange(0,n_layers):
        layer = Dense(n_neurons, activation=tf.nn.relu,kernel_initializer='random_uniform')#glorot_uniform is by default
        model.add(layer)

    # add layer to output array in correct dimensions
    output = Dense(N)
    model.add(output)

    # return finished model
    return model