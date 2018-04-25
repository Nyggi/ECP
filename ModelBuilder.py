from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, BatchNormalization, LeakyReLU, InputLayer
from keras.activations import linear
from keras.optimizers import Adam, RMSprop, sgd, Nadam, Adamax


class ModelBuilder:

    def __init__(self, cfg, input_shape):
        self.input_shape = input_shape
        self.activation_function = cfg.ACTIVATION_FUNCTION
        self.bias = cfg.BIAS
        self.loss = cfg.LOSS
        self.optimizer = cfg.OPTIMIZER
        self.hidden_layers = cfg.HIDDEN_LAYERS

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

        model.add(Dense(24, use_bias=self.bias))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["mape", "mae"])

        return model

    def nn_small(self):
        model = Sequential()

        model.add(Dense(24, use_bias=self.bias, input_shape=self.input_shape))
        model.add(Activation(self.activation_function))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["mape", "mae"])

        return model

    def nn_w(self):
        model = Sequential()

        model.add(InputLayer(input_shape=self.input_shape))

        for layer in self.hidden_layers:
            model.add(Dense(layer, use_bias=self.bias, activation=self.activation_function))

        model.add(Dense(24, use_bias=self.bias, activation=linear))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["mape", "mae"])

        return model
