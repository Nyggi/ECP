import mysql.connector
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.losses import mean_squared_error
from keras.models import clone_model
import json
from pandas import read_csv
from pandas import datetime

HourData = namedtuple('HourData', ['house_id', 'timestamp', 'consumption'])


# date-time parsing function for loading the dataset for energy consumption
def parser_energy(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))


def fetch_data(aggregation_level, group, hid):
    table = aggregation_level + "_" + group + "_energy_consumption"

    connection = mysql.connector.connect(host='localhost', user='root', password='root',
                                         database='households_energy_consumption')

    cursor = connection.cursor()  # Queries can be made through this Cursor object

    query = "SELECT house_id, timestamp, consumption  FROM " + table + " WHERE house_id = " + str(hid) + " ORDER BY timestamp;"

    cursor.execute(query)

    data = []

    for (house_id, timestamp, consumption) in cursor:
        # print("{}, {}, {}".format(house_id, str(timestamp), consumption))
        data.append(HourData(house_id, timestamp, consumption))

    return data


def fetch_data_csv(filename):
    raw_data = read_csv(filename, header=0, parse_dates=[0], index_col=0, squeeze=True,
                        date_parser=parser_energy, skiprows=0)
    data = list()
    for value in raw_data:
        data.append(HourData(1, 1, value))
    return data


def get_consumptions(data):
    consumptions = []

    for d in data:
        consumptions.append([d.consumption])

    return consumptions


def transform_data_for_nn(cfg, data):
    input_data = []
    targets = []
    extra_train = []
    extra_labels = []

    for q in range(cfg.DAYS * 24, len(data) - (24 * cfg.DAYS + cfg.HOURS_FUTURE)):
        l = []

        c = 0
        for i in range(q, q + 24 * cfg.DAYS):
            l.append(data[i][0])
            c += 1

        label = [data[q + 24 * cfg.DAYS + cfg.HOURS_FUTURE][0]]

        input_data.append(l)
        targets.append(label)

        if cfg.CRITICAL_START <= q % 24 <= cfg.CRITICAL_END:
            extra_train.append(l)
            extra_labels.append(label)

        day = q // 24

        if day % 7 == 0 or day % 7 == 5 or day % 7 == 6:
            if cfg.CRITICAL_START_WE <= q % 24 <= cfg.CRITICAL_END_WE:
                extra_train.append(l)
                extra_labels.append(label)

    return input_data, targets, extra_train, extra_labels


def evaluate(model, eval_input, eval_labels, graph_cut=1, show_graph=True):
    i = 0
    total_error = 0
    total_error_percent = 0

    max_error = 0

    predictions = []
    labels = []
    for inputD in eval_input:
        predict = np.array([inputD])
        p_1 = model.predict(predict)

        res = p_1[0][0]
        label = eval_labels[i][0]

        predictions.append(res)
        labels.append(label)

        error = abs(res - label)

        if error > max_error:
            max_error = error

        total_error_percent += error / label * 100
        total_error += error

        i += 1

    print("MAE:     " + f'{(total_error / i):.2f}')
    print("MAPE:    " + f'{(total_error_percent / i):.2f}')
    print("MAX:     " + f'{max_error:.2f}')

    if show_graph:
        line_up, = plt.plot(predictions[:int(len(predictions) * graph_cut)], label='Prediction')
        line_down, = plt.plot(labels[:int(len(predictions) * graph_cut)], label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.show(block=True)


def evaluate_freq(model, eval_input, eval_labels):
    i = 0
    errors = []

    for inputD in eval_input:
        predict = np.array([inputD])
        p_1 = model.predict(predict)

        res = p_1[0][0]
        label = eval_labels[i][0]

        error = (res - label) / label * 100
        errors.append(error)
        i += 1

    plot_error_freq(errors)
    plot_cumu_abs_error_freq(errors)
    plt.show()


def plot_error_freq(errors):
    bins = 25  # Odd number
    bin_min = -50
    bin_max = 50
    bin_diff = abs(bin_min - bin_max) / bins
    bins_list = [-10000]

    for i in range(bins - 1):
        bins_list.append(bin_min + (bin_diff * (i + 1)))

    bins_list.append(10000)

    weights = np.ones_like(errors) / float(len(errors)) * 100

    plt.hist(errors, bins=bins_list, weights=weights, edgecolor='black', linewidth=0.5)
    plt.xlim(xmin=bin_min, xmax=bin_max)
    plt.xlabel("Percentage error %")
    plt.ylabel("Frequency %")
    plt.xticks(np.arange(bin_min, bin_max + 1, step=10))
    plt.minorticks_on()
    plt.figure()


def plot_cumu_abs_error_freq(errors):
    abs_errors = [abs(x) for x in errors]

    bins = 13  # Odd number
    bin_min = -2
    bin_max = 50
    bin_diff = abs(bin_min - bin_max) / bins
    bins_list = [-10000]

    for i in range(bins - 1):
        bins_list.append(bin_min + (bin_diff * (i + 1)))

    bins_list.append(10000)

    weights = np.ones_like(errors) / float(len(errors)) * 100

    plt.hist(abs_errors, bins=bins_list, weights=weights, edgecolor='black', linewidth=0.5, cumulative=True)
    plt.xlim(xmin=0, xmax=bin_max)
    plt.xlabel("Cumulative Absolute Percentage error %")
    plt.ylabel("Frequency %")
    plt.xticks(np.arange(0, bin_max + 1, step=10))
    plt.minorticks_on()
    plt.figure()


def evaluate_other(cfg, model, aggregation_level, group, hid):
    other = fetch_data(aggregation_level, group, hid)

    other_v = get_consumptions(other)

    other_i, other_l, _, _ = transform_data_for_nn(cfg, other_v)

    print("-------------EVAL OTHER-------------------")

    evaluation = model.evaluate(np.array(other_i), np.array(other_l), batch_size=128)

    for thing in evaluation:
        print(thing)

    evaluate(model, other_i, other_l, cfg.GRAPH_CUT, show_graph=True)


def fit_model(cfg, model, train_input, train_labels):
    best_mape = 999999999

    best_model = clone_model(model)
    best_model.compile(loss=cfg.LOSS, optimizer=cfg.OPTIMIZER, metrics=["mape"])

    for e in range(cfg.EPOCHS):
        stats = model.fit(np.array(train_input), np.array(train_labels), cfg.BATCH_SIZE, 1, verbose=0, shuffle=True)
        mape = stats.history['mean_absolute_percentage_error'][0]

        print("Epoch " + str(e) + '/' + str(cfg.EPOCHS) + ' - MAPE: ' + str(mape))

        if mape < best_mape:
            best_mape = mape
            # Copy weigths from model to best model
            best_model.set_weights(model.get_weights())

    return best_model


def construct_training_data(cfg, data):
    sliced_data = data[:int(len(data) * cfg.DATA_SLICE)]

    values = get_consumptions(sliced_data)

    input_data, labels, extra_train, extra_labels = transform_data_for_nn(cfg, values)

    validation_cut = int(len(input_data) * cfg.TRAINING_CUT)

    train_input = input_data[0:validation_cut]
    train_input.extend(extra_train)

    train_labels = labels[0:validation_cut]
    train_labels.extend(extra_labels)

    eval_input = input_data[validation_cut:]
    eval_labels = labels[validation_cut:]

    return train_input, train_labels, eval_input, eval_labels


def weight_mmma(model):
    weights = model.get_weights()

    layers = []

    for layer in weights:
        nodes = []
        for node in layer:
            weight_min = min(node)
            weight_max = max(node)
            median = node[len(node) // 2]
            average = sum(node)/len(node)

            weight_string = f'MIN {weight_min:.4f}, MAX {weight_max:.4f}, Median {median:.4f}, AVG {average:.4f}'
            nodes.append(weight_string)

        layers.append(nodes)

    with open('weights_mmma.json', 'w') as outfile:
        json_string = json.dumps(layers)
        outfile.write(json_string)

