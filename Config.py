from random import uniform, randrange
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
        self.REMOVE_OUTLIERS = True

        self.TRAINING_CUT = 0.7
        self.DATA_SLICE = 1
        self.GRAPH_CUT = 1

        # Features
        # Same hours in past days past weeks
        # X hours past
        # Day of week
        self.FEATURES = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.PADDING = 0

        #If WEKA_FEATURES is True, FEATURES will not be used.
        self.WEKA_FEATURES = True
        self.WRITE_CSV = False
        self.WEKA_MULTIPLE_HOUSEHOLDS = False
