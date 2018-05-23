import EvalMetrics
import mysql.connector
from collections import namedtuple
import numpy as np

HourData = namedtuple('HourData', ['house_id', 'timestamp', 'consumption'])


class BaselineModel:

    def __init__(self, house_id):
        self.house_id = house_id
        self.data = self.fetch_data()

    def fetch_data(self):
        connection = mysql.connector.connect(host='localhost', user='root', password='root',
                                             database='households_energy_consumption')

        cursor = connection.cursor()  # Queries can be made through this Cursor object

        query = 'SELECT house_id, timestamp, consumption  FROM hourly_households_energy_consumption WHERE house_id = ' + str(self.house_id) + ' ORDER BY timestamp;'

        cursor.execute(query)

        data = []

        for (house_id, timestamp, consumption) in cursor:
            # print("{}, {}, {}".format(house_id, str(timestamp), consumption))
            data.append(HourData(house_id, timestamp, consumption))

        return data

    def eval_model(self):
        metrics = [EvalMetrics.mape, EvalMetrics.mer, EvalMetrics.mae]

        predictions_list = []
        labels_list = []

        for i in range(24):
            predictions_list.append([])
            labels_list.append([])

        start = int(0.7 * len(self.data))

        for hour in range(start + (24 * 7), len(self.data)):
            predictions_list[hour % 24].append([self.predict(hour)])
            labels_list[hour % 24].append([self.data[hour].consumption])

        all_eval_values = []

        for i in range(24):
            eval_values = []

            for metric in metrics:
                eval_values.append(metric(np.array(predictions_list[i]), np.array(labels_list[i])))

            all_eval_values.append(eval_values)

        return all_eval_values

    def predict(self, hour):
        if hour - 24 < 0:
            raise Exception()

        #prediction = self.data[hour - 24].consumption
        prediction = self.data[hour - 24 * 7].consumption

        return prediction
