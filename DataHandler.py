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
        input_data = []
        targets = []

        for q in range(self.cfg.DAYS * 24, len(data) - (24 * self.cfg.DAYS + self.cfg.HOURS_FUTURE)):
            l = []

            c = 0
            for i in range(q, q + 24 * self.cfg.DAYS):
                l.append(data[i])
                c += 1

            label = data[q + 24 * self.cfg.DAYS + self.cfg.HOURS_FUTURE]

            input_data.append(l)
            targets.append(label)

        return input_data, targets

    def _construct_training_data(self, data):
        sliced_data = data[:int(len(data) * self.cfg.DATA_SLICE)]

        values = self._get_consumptions(sliced_data)

        input_data, labels = self._transform_data_for_nn(values)

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
