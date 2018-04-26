import mysql.connector
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
import numpy as np

HourData = namedtuple('HourData', ['house_id', 'timestamp', 'consumption'])


class DataHandler:

    def __init__(self, cfg, house_id):
        self.cfg = cfg
        self.house_id = house_id
        self.data = self._fetch_data(house_id)

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if cfg.SCALE_VALUES is True:
            scaled_data = self._get_scaled_data()
            train_input, train_labels, eval_input, eval_labels = self._construct_training_data(scaled_data)
        else:
            self.scaler = None
            train_input, train_labels, eval_input, eval_labels = self._construct_training_data(self.data)

        self.train_input = train_input
        self.train_labels = train_labels
        self.eval_input = eval_input
        self.eval_labels = eval_labels

    def _fetch_data(self, house_id):
        connection = mysql.connector.connect(host='localhost', user='root', password='root',
                                             database='households_energy_consumption')

        cursor = connection.cursor()  # Queries can be made through this Cursor object

        query = 'SELECT house_id, timestamp, consumption  FROM hourly_households_energy_consumption WHERE house_id = ' + str(house_id) + ' ORDER BY timestamp;'

        cursor.execute(query)

        data = []

        for (house_id, timestamp, consumption) in cursor:
            # print("{}, {}, {}".format(house_id, str(timestamp), consumption))
            data.append(HourData(house_id, timestamp, consumption))

        return data

    def _extract_features(self, data):
        X = []
        y = []

        past_hours_padding = self.cfg.HOURS_PAST + 12 + self.cfg.HOUR_TO_PREDICT
        past_days_padding = self.cfg.WEEKS * 24 * 7

        if past_days_padding > past_hours_padding:
            padding = past_days_padding
        else:
            padding = past_hours_padding

        padding += 24 - padding % 24

        for label in range(padding + self.cfg.HOUR_TO_PREDICT, len(data), 24):
            features = []

            # Same hour in same day in past weeks
            if self.cfg.FEATURES[0]:
                for w in range(1, self.cfg.WEEKS + 1):
                    hour = label - (w * 7 * 24)
                    features.append(data[hour].consumption)

            # Past hours
            if self.cfg.FEATURES[1]:
                for h in range(self.cfg.HOURS_PAST):
                    hour = label - (12 + h + self.cfg.HOUR_TO_PREDICT)
                    features.append(data[hour].consumption)

            # Binary encoding of day of the week
            if self.cfg.FEATURES[2]:
                weekday = data[label].timestamp.weekday()
                weekday_bin = f'{weekday:03b}'
                for b in weekday_bin:
                    features.append(int(b))

            X.append(features)
            y.append(data[label].consumption)

        return X, y

    def _construct_training_data(self, data):
        sliced_data = data[:int(len(data) * self.cfg.DATA_SLICE)]

        input_data, labels = self._extract_features(sliced_data)

        validation_cut = int(len(input_data) * self.cfg.TRAINING_CUT)

        train_input = input_data[0:validation_cut]
        train_labels = labels[0:validation_cut]

        eval_input = input_data[validation_cut:]
        eval_labels = labels[validation_cut:]

        return train_input, train_labels, eval_input, eval_labels

    def _get_scaled_data(self):
        consumption_list = list()

        for i in self.data:
            consumption_list.append([i.consumption])

        consumption_list = np.asarray(consumption_list)
        scaled_consumption = self.scaler.fit_transform(consumption_list)

        scaled_data = list()

        for j in range(len(self.data)):
            scaled_data.append(HourData(self.data[j].house_id, self.data[j].timestamp, scaled_consumption[j][0]))

        return scaled_data
