from keras.losses import *
from keras.optimizers import *


class Config:
    def __init__(self):
        self.EPOCHS = 60
        self.BATCH_SIZE = 128

        self.HOUSE_ID = 5

        # 0 - 23
        self.HOUR_TO_PREDICT = 17

        self.HOURS_PAST = 48
        self.WEEKS = 5
        self.DAYS = 5

        self.HIDDEN_LAYERS = [100]

        self.BIAS = True
        self.ACTIVATION_FUNCTION = 'tanh'
        self.LOSS = mean_squared_error
        self.OPTIMIZER = Adamax()
        self.SCALE_VALUES = True
        self.SCALE_RANGE = (-1, 1)

        self.TRAINING_CUT = 0.7
        self.DATA_SLICE = 1
        self.GRAPH_CUT = 1

        # Features
        # Same hours in past days past weeks
        # X hours past
        # Day of week
        self.FEATURES = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.FEATURES_BINARY_ENCODED = False
        self.PADDING = 0
