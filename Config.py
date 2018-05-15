from keras import losses
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

        self.HIDDEN_LAYERS = [100, 24, 4]

        self.BIAS = True
        self.ACTIVATION_FUNCTION = 'tanh'
        self.LOSS = losses.mean_absolute_error
        self.OPTIMIZER = Adamax()
        self.SCALE_VALUES = True
        self.SCALE_RANGE = (-1, 1)
        self.REMOVE_OUTLIERS = True

        self.TRAINING_CUT = 0.7
        self.DATA_SLICE = 1
        self.GRAPH_CUT = 1
        self.SHUFFLE = False

        # Features
        # Same hours in past days past weeks
        # X hours past
        # Day of week
        self.FEATURES = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        self.PADDING = 0

        # If WEKA_FEATURES is True, FEATURES will not be used.
        self.WEKA_FEATURES = True
        self.SMF_FEATURES = False
        self.WEKA_HOUSEHOLD_IDS = [5]
