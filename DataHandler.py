import mysql.connector
from collections import namedtuple

HourData = namedtuple('HourData', ['house_id', 'timestamp', 'consumption'])


class DataHandler:

    def __init__(self, cfg,  house_id):
        self.cfg = cfg
        self.house_id = house_id
        self.data = self._fetch_data(house_id)

        train_input, train_labels, eval_input, eval_labels = self._construct_training_data(self.data)

        self.train_input = train_input
        self.train_labels = train_labels
        self.eval_input = eval_input
        self.eval_labels = eval_labels

    def _fetch_data(self, house_id):
        connection = mysql.connector.connect(host='localhost', user='root', password='root',
                                             database='households_energy_consumption')

        cursor = connection.cursor()  # Queries can be made through this Cursor object

        query = 'SELECT house_id, timestamp, consumption  FROM hourly_households_energy_consumption WHERE house_id = ' + str(house_id) +' ORDER BY timestamp;'

        cursor.execute(query)

        data = []

        for (house_id, timestamp, consumption) in cursor:
            # print("{}, {}, {}".format(house_id, str(timestamp), consumption))
            data.append(HourData(house_id, timestamp, consumption))

        return data

    def _transform_data_for_nn(self, data):
        X = []
        y = []

        past_hours_padding = self.cfg.HOURS_PAST + self.cfg.HOURS_FUTURE + 24
        past_days_padding = self.cfg.WEEKS * 24 * 7 + self.cfg.HOURS_FUTURE + 24

        if past_days_padding > past_hours_padding:
            padding = past_days_padding
        else:
            padding = past_hours_padding

        for label in range(padding, len(data) - self.cfg.HOURS_FUTURE - 24):
            features = []

            # Same hours in same day in past weeks
            if self.cfg.FEATURES[0]:
                for w in range(1, self.cfg.WEEKS + 1):
                    for h in range(label - w * 7 * 24, label - w * 7 * 24 + 24):
                        features.append(data[h].consumption)

            # Past hours
            if self.cfg.FEATURES[1]:
                for h in range(self.cfg.HOURS_PAST):
                    features.append(data[label - h].consumption)

            # Binary encoding of day of the week
            if self.cfg.FEATURES[2]:
                weekday = data[label].timestamp.weekday()
                weekday_bin = f'{weekday:03b}'
                for b in weekday_bin:
                    features.append(int(b))

            X.append(features)

            l = []

            for hour in range(24):
                l.append(data[label + hour].consumption)

            y.append(l)

        return X, y

    def _construct_training_data(self, data):
        sliced_data = data[:int(len(data) * self.cfg.DATA_SLICE)]

        input_data, labels = self._transform_data_for_nn(sliced_data)

        validation_cut = int(len(input_data) * self.cfg.TRAINING_CUT)

        train_input = input_data[0:validation_cut]
        train_labels = labels[0:validation_cut]

        eval_input = input_data[validation_cut:]
        eval_labels = labels[validation_cut:]

        return train_input, train_labels, eval_input, eval_labels

    def _get_consumptions(self, data):
        consumptions = []

        for d in data:
            consumptions.append(d.consumption)

        return consumptions
