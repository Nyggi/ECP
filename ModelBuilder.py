from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, BatchNormalization, LeakyReLU, Input
from keras.optimizers import Adam, RMSprop, sgd, Nadam, Adamax
from random import randrange, random


class ModelBuilder:

    def __init__(self, cfg, input_shape):
        self.input_shape = input_shape
        self.activation_function = cfg.ACTIVATION_FUNCTION
        self.bias = cfg.BIAS
        self.loss = cfg.LOSS
        self.optimizer = cfg.OPTIMIZER

    def nn(self):
        model = Sequential()

        model.add(Dense(1024, use_bias=self.bias, input_shape=self.input_shape))
        model.add(Activation(self.activation_function))

        model.add(Dense(512, use_bias=self.bias))
        model.add(Activation(self.activation_function))

        model.add(Dense(256, use_bias=self.bias))
        model.add(Activation(self.activation_function))

        model.add(Dense(128, use_bias=self.bias))
        model.add(Activation(self.activation_function))

        model.add(Dense(16, use_bias=self.bias))
        model.add(Activation(self.activation_function))

        model.add(Dense(8, use_bias=self.bias))
        model.add(Activation(self.activation_function))

        model.add(Dense(1, use_bias=self.bias))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["mape"])

        return model

    def nn_small(self):
        model = Sequential()

        model.add(Dense(128, use_bias=self.bias, input_shape=self.input_shape))
        model.add(Activation(self.activation_function))

        model.add(Dense(1, use_bias=self.bias))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["mape"])

        return model

    def nn_gen(self):
        model = Sequential()

        layers = randrange(3, 20)

        rn = randrange(100, 5000, 10)

        model.add(Dense(rn, use_bias=self.bias, input_shape=self.input_shape))

        for i in range(layers, 1):
            if random.random() < 0.7:
                nodes = randrange(10 * i, 200 * i)
                model.add(Dense(nodes, use_bias=self.bias))
                model.add(Activation(self.activation_function))

        model.add(Dense(1, use_bias=self.bias))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["mape"])

        return model

    def nn_lin(self):
        model = Sequential()

        model.add(Dense(1, use_bias=self.bias, input_shape=self.input_shape))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["mape"])

        return model
