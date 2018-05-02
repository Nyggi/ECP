import matplotlib.pyplot as plt
import pandas
import numpy as np
import mysql.connector
from collections import namedtuple

COMPARE_SIZE = 840

HOUSE = 5

SELECT_SPECIFIC = True

HOURS_PAST = 48
DAYS = 21
WEEKS = 15

FEATURES = []

HourData = namedtuple('HourData', ['house_id', 'timestamp', 'consumption'])

for i in range(24):
    open("PandasData" + str(i) + ".csv", 'w').close()

def create_features_list():
    for i in range(DAYS):
        FEATURES.append(48 + (i * 24))

    for i in range(WEEKS):
        FEATURES.append((i + 1) * 168)


def fetch_data_hourly():
    data = []

    connection = mysql.connector.connect(host='localhost', user='root', password='root',
                                         database='households_energy_consumption')

    cursor = connection.cursor()  #Queries can be made through this Cursor object

    housedata = []

    query = 'SELECT house_id, timestamp, consumption FROM hourly_households_energy_consumption ' \
            'WHERE house_id = ' + str(HOUSE) + ' ORDER BY timestamp;'

    cursor.execute(query)

    for (house_id, timestamp, consumption) in cursor:
        # print("{}, {}, {}".format(house_id, str(timestamp), consumption))
        housedata.append(HourData(house_id, timestamp, consumption))

    data.append(housedata)

    return data


def prepare_data_for_pandas(consumption_data):
    for house in consumption_data:
        for j in range(24):
            padding = j + 12

            label_list = []

            for i in range(1, HOURS_PAST + 1, 1):
                label_list.append(i + padding)


            label_list = label_list + FEATURES

            label_list = list(sorted(set(label_list)))

            if SELECT_SPECIFIC:
                with open('PandasData' + str(j) + '.csv', 'a+') as f:

                    for i in label_list:
                        f.write("_" + str(i))
                        f.write(",")

                    f.write("label")
                    f.write("\n")

                    for i in range(label_list[-1] + 1, len(house), 1):
                        if (house[i].timestamp.hour == j + 1) or (j == 23 and house[i].timestamp.hour == 0):
                            for input in label_list:
                                f.write(str(house[i - input].consumption))
                                f.write(",")

                            f.write(str(house[i].consumption))
                            f.write("\n")

            else:
                with open('PandasData' + str(j) + '.csv', 'a+') as f:
                    #Disable this for no names in csv
                    #Counts from COMPARE_SIZE +1 to and including 1 in decreasing order and writes it on a like in the csv.
                    for l in range(COMPARE_SIZE, 0, -1):
                        f.write(str(l))
                        f.write(",")
                    #Disable this for no names in csv
                    f.write("label")
                    f.write("\n")

                    #Loops from the last hour to predict in the data, towards the first hour to predict in the data.
                    #j + 12 ensures that we don't go out of bounds because of the required COMPARE_SIZE.
                    for i in range(len(house) - 1, COMPARE_SIZE + padding, -1):
                        #Ensures that the hour we are to predict is the correct hour, since all predictions of the
                        #same hour is gathered in the same file.
                        if (house[i].timestamp.hour == j+1) or (j == 23 and house[i].timestamp.hour == 0):
                            #Loops from the input hour furthest away from the output label, to the input hour closest to the output label.
                            for k in range(i - COMPARE_SIZE - padding, i - padding, 1):
                                f.write(str(house[k].consumption))
                                f.write(",")

                                #If it is the last input hour to write to the file, it appends it with the output label and a newline.
                                if k == (i - padding - 1):
                                    f.write(str(house[i].consumption))
                                    f.write("\n")


create_features_list()
data = fetch_data_hourly()
prepare_data_for_pandas(data)

