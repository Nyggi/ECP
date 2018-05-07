import mysql.connector
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
import numpy as np

HourData = namedtuple('HourData', ['house_id', 'timestamp', 'consumption'])


class DataHandler:

    def __init__(self, cfg):
        self.cfg = cfg
        self.house_id = cfg.HOUSE_ID
        self.data = self._fetch_data(self.house_id)

        self.scaler = MinMaxScaler(feature_range=cfg.SCALE_RANGE)

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
                    for h in range(hour - self.cfg.PADDING, hour + 1 + self.cfg.PADDING):
                        features.append(data[hour].consumption)

            # Same hour in same day in past weeks STANDARD DEVIATION
            if self.cfg.FEATURES[1]:
                hours = []
                for w in range(1, self.cfg.WEEKS + 1):
                    hour = label - (w * 7 * 24)
                    hours.append(data[hour].consumption)

                sd = np.std(hours)
                features.append(sd)

            # Same hour in same day in past weeks MIN, MEAN, MAX
            if self.cfg.FEATURES[2]:
                hours = []
                for w in range(1, self.cfg.WEEKS + 1):
                    hour = label - (w * 7 * 24)
                    hours.append(data[hour].consumption)

                features.append(max(hours))
                features.append(min(hours))
                features.append(np.mean(hours))

            # Same hour in past days
            if self.cfg.FEATURES[3]:
                for d in range(1, self.cfg.DAYS + 1):
                    hour = label - ((d + 1) * 24)
                    for h in range(hour - self.cfg.PADDING, hour + 1 + self.cfg.PADDING):
                        features.append(data[hour].consumption)

            # Same hour in past days STANDARD DEVIATION
            if self.cfg.FEATURES[4]:
                hours = []
                for d in range(1, self.cfg.DAYS + 1):
                    hour = label - ((d + 1) * 24)
                    hours.append(data[hour].consumption)

                sd = np.std(hours)
                features.append(sd)

            # Same hour in past days MIN, MEAN, MAX
            if self.cfg.FEATURES[5]:
                hours = []
                for d in range(1, self.cfg.DAYS + 1):
                    hour = label - ((d + 1) * 24)
                    hours.append(data[hour].consumption)

                features.append(max(hours))
                features.append(min(hours))
                features.append(np.mean(hours))

            # Past hours
            if self.cfg.FEATURES[6]:
                for h in range(self.cfg.HOURS_PAST):
                    hour = label - (12 + h + self.cfg.HOUR_TO_PREDICT)
                    features.append(data[hour].consumption)

            # Time of day
            if self.cfg.FEATURES[7]:
                hour = data[label].timestamp.hour
                if self.cfg.FEATURES_BINARY_ENCODED:
                    hour_bin = f'{hour:05b}'
                    for b in hour_bin:
                        features.append(int(b))
                else:
                    hour_rad = hour * (2. * np.pi / 24)
                    hour_sin = np.sin(hour_rad)
                    hour_cos = np.cos(hour_rad)
                    features.append(hour_sin)
                    features.append(hour_cos)

            # Day of the week
            if self.cfg.FEATURES[8]:
                weekday = data[label].timestamp.weekday()
                if self.cfg.FEATURES_BINARY_ENCODED:
                    weekday_bin = f'{weekday:03b}'
                    for b in weekday_bin:
                        features.append(int(b))
                else:
                    weekday_rad = weekday * (2. * np.pi / 7)
                    weekday_sin = np.sin(weekday_rad)
                    weekday_cos = np.cos(weekday_rad)
                    features.append(weekday_sin)
                    features.append(weekday_cos)

            # Month of year
            if self.cfg.FEATURES[9]:
                month = data[label].timestamp.month
                if self.cfg.FEATURES_BINARY_ENCODED:
                    month_bin = f'{month:04b}'
                    for b in month_bin:
                        features.append(int(b))
                else:
                    month_rad = month * (2. * np.pi / 12)
                    month_sin = np.sin(month_rad)
                    month_cos = np.cos(month_rad)
                    features.append(month_sin)
                    features.append(month_cos)

            # Season of year - 0 Winter ... 3 Autumn
            if self.cfg.FEATURES[10]:
                month = data[label].timestamp.month
                season = int((month + 1) / 3) % 4
                if self.cfg.FEATURES_BINARY_ENCODED:
                    season_bin = f'{season:02b}'
                    for b in season_bin:
                        features.append(int(b))
                else:
                    season_rad = season * (2. * np.pi / 12)
                    season_sin = np.sin(season_rad)
                    season_cos = np.cos(season_rad)
                    features.append(season_sin)
                    features.append(season_cos)

            # Holiday
            if self.cfg.FEATURES[11]:
                weekday = data[label].timestamp.weekday()
                if weekday == 5 or weekday == 6:
                    features.append(1)
                else:
                    features.append(0)

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
            consumption_list.append([float(i.consumption)])

        consumption_list = np.asarray(consumption_list)
        scaled_consumption = self.scaler.fit_transform(consumption_list)

        scaled_data = list()

        for j in range(len(self.data)):
            scaled_data.append(HourData(self.data[j].house_id, self.data[j].timestamp, scaled_consumption[j][0]))

        return scaled_data
