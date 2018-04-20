import matplotlib.pyplot as plt
import pandas
import numpy as np
import mysql.connector
from collections import namedtuple

COMPARE_SIZE = 336

HourData = namedtuple('HourData', ['house_id', 'timestamp', 'consumption'])
open("PandasData.csv", 'w').close()


def fetch_data_hourly():
    data = []

    connection = mysql.connector.connect(host='localhost', user='root', password='root',
                                         database='households_energy_consumption')

    cursor = connection.cursor()  # Queries can be made through this Cursor object

    query = 'SELECT COUNT(DISTINCT(house_id)) ' \
            'FROM households_energy_consumption.hourly_multiple_households_energy_consumption;'

    cursor.execute(query)

    amount_of_houses = cursor.fetchall()[0][0]

    #Choose range(amount_of_houses) if you want stats for all households in data instead of 1.
    for i in range(1):
        housedata = []

        query = 'SELECT house_id, timestamp, consumption FROM hourly_multiple_households_energy_consumption ' \
                'WHERE house_id = ' + str(i) + ' ORDER BY timestamp;'

        cursor.execute(query)

        for (house_id, timestamp, consumption) in cursor:
            # print("{}, {}, {}".format(house_id, str(timestamp), consumption))
            housedata.append(HourData(house_id, timestamp, consumption))

        data.append(housedata)

    return data


def fetch_data_daily():
    data = []

    connection = mysql.connector.connect(host='localhost', user='root', password='root',
                                         database='households_energy_consumption')

    cursor = connection.cursor()  # Queries can be made through this Cursor object

    query = 'SELECT COUNT(DISTINCT(house_id)) ' \
            'FROM households_energy_consumption.hourly_multiple_households_energy_consumption;'

    cursor.execute(query)

    amount_of_houses = cursor.fetchall()[0][0]

    # Choose range(amount_of_houses) if you want stats for all households in data instead of 1.
    for i in range(1):
        c = 0
        summed_consumption = 0
        housedata = []

        query = 'SELECT house_id, timestamp, consumption FROM hourly_multiple_households_energy_consumption ' \
                'WHERE house_id = ' + str(i) + ' ORDER BY timestamp;'

        cursor.execute(query)

        for (house_id, timestamp, consumption) in cursor:
            # print("{}, {}, {}".format(house_id, str(timestamp), consumption))
            summed_consumption += consumption

            if c % 24 == 0 and c != 0:
                housedata.append(HourData(house_id, timestamp, summed_consumption))
                summed_consumption = 0

            c += 1

        data.append(housedata)

    return data


def make_correlation_matrix(consumption_data):
    path = "PandasData.csv"
    names = []

    for hour in range(COMPARE_SIZE):
        names.append(str(hour))

    data = pandas.read_csv(path, names=names)
    correlations = data.corr()
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, COMPARE_SIZE, 24)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()


def prepare_data_for_pandas(consumption_data):
    f = open("PandasData.csv", "a+")

    for house in consumption_data:
        c_size = 0

        for i in range(len(house) - COMPARE_SIZE):
            for j in range(COMPARE_SIZE):
                f.write(str(house[i + j].consumption))

                if j != COMPARE_SIZE - 1:
                    f.write(",")

                if j == COMPARE_SIZE - 1:
                    f.write("\n")

            c_size += 1


data = fetch_data_hourly()
# data = fetch_data_daily()
prepare_data_for_pandas(data)
make_correlation_matrix(data)
