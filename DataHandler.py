import mysql.connector
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from random import shuffle


HourData = namedtuple('HourData', ['house_id', 'timestamp', 'consumption'])


class DataHandler:

    def __init__(self, cfg):
        self.cfg = cfg
        self.data = self._fetch_data(self.cfg.HOUSE_ID)

        if cfg.REMOVE_OUTLIERS is True:
            data_to_continue = self._remove_outliers(0.99, 80000)
        else:
            data_to_continue = self.data

        self.scaler = MinMaxScaler(feature_range=cfg.SCALE_RANGE)

        if cfg.SCALE_VALUES is True:
            scaled_data = self._get_scaled_data(data_to_continue)
            train_input, train_labels, eval_input, eval_labels = self._construct_training_data(scaled_data)
        else:
            self.scaler = None
            train_input, train_labels, eval_input, eval_labels = self._construct_training_data(data_to_continue)

        self.train_input = train_input
        self.train_labels = train_labels
        self.eval_input = eval_input
        self.eval_labels = eval_labels

    def _get_csv_data(self):
        features = self.train_input + self.eval_input
        results = self.train_labels + self.eval_labels

        content_for_csv = ''

        for i in range(len(features)):
            for j in features[i]:
                content_for_csv += str(j)
                content_for_csv += ','
            content_for_csv += str(results[i])
            content_for_csv += '\n'

        return content_for_csv

    def _get_csv_header(self):
        past_hours_padding = self.cfg.HOURS_PAST + 12 + self.cfg.HOUR_TO_PREDICT
        past_days_padding = self.cfg.WEEKS * 24 * 7

        if past_days_padding > past_hours_padding:
            padding = past_days_padding
        else:
            padding = past_hours_padding

        padding += 24 - padding % 24

        header = ""

        # Same hour in same day in past weeks
        if self.cfg.FEATURES[0]:
            for w in range(1, self.cfg.WEEKS + 1):
                hour = (w * 7 * 24)
                for h in range(hour - self.cfg.PADDING, hour + 1 + self.cfg.PADDING):
                    header += '0_' + str(h) + ','

        # Same hour in same day in past weeks STANDARD DEVIATION
        if self.cfg.FEATURES[1]:
            header += '1_0,'

        # Same hour in same day in past weeks MIN, MEAN, MAX
        if self.cfg.FEATURES[2]:
            header += '2_0,'
            header += '2_1,'
            header += '2_2,'

        # Same hour in past days
        if self.cfg.FEATURES[3]:
            for d in range(1, self.cfg.DAYS + 1):
                hour = ((d + 1) * 24)
                for h in range(hour - self.cfg.PADDING, hour + 1 + self.cfg.PADDING):
                    header += '3_' + str(h) + ','

        # Same hour in past days STANDARD DEVIATION
        if self.cfg.FEATURES[4]:
            header += '4_0,'

        # Same hour in past days MIN, MEAN, MAX
        if self.cfg.FEATURES[5]:
            header += '5_0,'
            header += '5_1,'
            header += '5_2,'

        # Past hours
        if self.cfg.FEATURES[6]:
            for h in range(self.cfg.HOURS_PAST):
                hour = (12 + h + self.cfg.HOUR_TO_PREDICT)
                header += '6_' + str(hour) + ','

        # Day of the week
        if self.cfg.FEATURES[7]:
            header += '7_0,'
            header += '7_1,'

        # Month of year
        if self.cfg.FEATURES[8]:
            header += '8_0,'
            header += '8_1,'

        # Season of year - 0 Winter ... 3 Autumn
        if self.cfg.FEATURES[9]:
            header += '9_0,'
            header += '9_1,'

        # Holiday
        if self.cfg.FEATURES[10]:
            header += '10_0,'

        header += 'label\n'

        return header

    def _extract_features_weka(self, data):
        X = []
        y = []

        past_hours_padding = self.cfg.HOURS_PAST + 12 + self.cfg.HOUR_TO_PREDICT
        past_days_padding = self.cfg.WEEKS * 24 * 7

        if past_days_padding > past_hours_padding:
            padding = past_days_padding
        else:
            padding = past_hours_padding

        padding += 24 - padding % 24

        if len(self.cfg.WEKA_HOUSEHOLD_IDS) > 1:
            amount_of_households = 'multiple'
        else:
            amount_of_households = 'single'

        if self.cfg.SMF:
            filepath = 'WEKA_features/best_features_from_WEKA_SMF.csv'
        else:
            filepath = 'WEKA_features/best_features_from_WEKA_' + str(amount_of_households) + '/BestFeatures' + str(self.cfg.HOUR_TO_PREDICT) + '.csv'

        csvfile = open(filepath, 'r')

        for line in csvfile:
            csv_line = line

        csv_features = csv_line.split(',')
        csvfile.close()

        for label in range(padding + self.cfg.HOUR_TO_PREDICT, len(data), 24):
            features = []

            for feature in csv_features:
                splitted_feature = str(feature).split('_')
                index = splitted_feature[0]
                value = splitted_feature[1]

                # Same hour in same day in past weeks OR
                # Same hour in past days OR
                # Past hours
                if index == '0' or index == '3' or index == '6':
                    features.append(data[label - int(value)].consumption)

                # Same hour in same day in past weeks STANDARD DEVIATION
                if index == '1':
                    hours = []
                    for w in range(1, self.cfg.WEEKS + 1):
                        hour = label - (w * 7 * 24)
                        hours.append(data[hour].consumption)

                    sd = np.std(hours)
                    features.append(sd)

                # Same hour in same day in past weeks MIN, MEAN, MAX
                if index == '2':
                    hours = []
                    for w in range(1, self.cfg.WEEKS + 1):
                        hour = label - (w * 7 * 24)
                        hours.append(data[hour].consumption)

                    if value == '0':
                        features.append(max(hours))
                    if value == '1':
                        features.append(min(hours))
                    if value == '2':
                        features.append(np.mean(hours))

                # Same hour in past days STANDARD DEVIATION
                if index == '4':
                    hours = []
                    for d in range(1, self.cfg.DAYS + 1):
                        hour = label - ((d + 1) * 24)
                        hours.append(data[hour].consumption)

                    sd = np.std(hours)
                    features.append(sd)

                # Same hour in past days MIN, MEAN, MAX
                if index == '5':
                    hours = []
                    for d in range(1, self.cfg.DAYS + 1):
                        hour = label - ((d + 1) * 24)
                        hours.append(data[hour].consumption)

                    if value == '0':
                        features.append(max(hours))
                    if value == '1':
                        features.append(min(hours))
                    if value == '2':
                        features.append(np.mean(hours))

                # Day of the week
                if index == '7':
                    weekday = data[label].timestamp.weekday()
                    weekday_rad = weekday * (2. * np.pi / 7)
                    weekday_sin = np.sin(weekday_rad)
                    weekday_cos = np.cos(weekday_rad)

                    if value == '0':
                        features.append(weekday_sin)
                    if value == '1':
                        features.append(weekday_cos)

                # Month of year
                if index == '8':
                    month = data[label].timestamp.month
                    month_rad = month * (2. * np.pi / 12)
                    month_sin = np.sin(month_rad)
                    month_cos = np.cos(month_rad)

                    if value == '0':
                        features.append(month_sin)
                    if value == '1':
                        features.append(month_cos)

                # Season of year - 0 Winter ... 3 Autumn
                if index == '9':
                    month = data[label].timestamp.month
                    season = int((month + 1) / 3) % 4
                    season_rad = season * (2. * np.pi / 12)
                    season_sin = np.sin(season_rad)
                    season_cos = np.cos(season_rad)

                    if value == '0':
                        features.append(season_sin)
                    if value == '1':
                        features.append(season_cos)

                # Holiday
                if index == '10':
                    weekday = data[label].timestamp.weekday()
                    if weekday == 5 or weekday == 6:
                        features.append(1)
                    else:
                        features.append(0)

            X.append(features)
            y.append(data[label].consumption)

        return X, y

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

    def _remove_outliers(self, percentile, outlier_min):
        percentile_down_to = sorted(self.data, key=lambda tup: tup[2])[int(len(self.data) * percentile)].consumption

        new_data = []

        for d in self.data:
            if d.consumption > outlier_min:
                new_data.append(HourData(d.house_id, d.timestamp, percentile_down_to))
            else:
                new_data.append(HourData(d.house_id, d.timestamp, d.consumption))

        return new_data

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

            # Day of the week
            if self.cfg.FEATURES[7]:
                weekday = data[label].timestamp.weekday()
                weekday_rad = weekday * (2. * np.pi / 7)
                weekday_sin = np.sin(weekday_rad)
                weekday_cos = np.cos(weekday_rad)
                features.append(weekday_sin)
                features.append(weekday_cos)

            # Month of year
            if self.cfg.FEATURES[8]:
                month = data[label].timestamp.month
                month_rad = month * (2. * np.pi / 12)
                month_sin = np.sin(month_rad)
                month_cos = np.cos(month_rad)
                features.append(month_sin)
                features.append(month_cos)

            # Season of year - 0 Winter ... 3 Autumn
            if self.cfg.FEATURES[9]:
                month = data[label].timestamp.month
                season = int((month + 1) / 3) % 4
                season_rad = season * (2. * np.pi / 12)
                season_sin = np.sin(season_rad)
                season_cos = np.cos(season_rad)
                features.append(season_sin)
                features.append(season_cos)

            # Holiday
            if self.cfg.FEATURES[10]:
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

        if self.cfg.WEKA_FEATURES:
            input_data, labels = self._extract_features_weka(sliced_data)
        else:
            input_data, labels = self._extract_features(sliced_data)

        if self.cfg.SHUFFLE:
            shuffled_data = []

            for i in range(len(input_data)):
                shuffled_data.append([input_data[i], labels[i]])

            shuffle(shuffled_data)

            input_data_shuffled = []
            labels_shuffled = []

            for i in range(len(input_data)):
                input_data_shuffled.append(shuffled_data[i][0])
                labels_shuffled.append(shuffled_data[i][1])

            input_data = input_data_shuffled
            labels = labels_shuffled

        validation_cut = int(len(input_data) * self.cfg.TRAINING_CUT)

        train_input = input_data[0:validation_cut]
        train_labels = labels[0:validation_cut]

        eval_input = input_data[validation_cut:]
        eval_labels = labels[validation_cut:]

        return train_input, train_labels, eval_input, eval_labels

    def _get_scaled_data(self, data):
        consumption_list = list()

        for i in data:
            consumption_list.append([float(i.consumption)])

        consumption_list = np.asarray(consumption_list)
        scaled_consumption = self.scaler.fit_transform(consumption_list)

        scaled_data = list()

        for j in range(len(data)):
            scaled_data.append(HourData(data[j].house_id, data[j].timestamp, scaled_consumption[j][0]))

        return scaled_data
