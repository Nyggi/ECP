import mysql.connector
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.losses import mean_squared_error
from keras.models import clone_model

HourData = namedtuple('HourData', ['house_id', 'timestamp', 'consumption'])


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

    intervals = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 1000]

    i = 0
    freq = [0 for k in range(len(intervals))]
    entries = len(errors)
    for p in intervals:
        for e in errors:
            if -p < e < p:
                freq[i] += 1
                errors.remove(e)
        i += 1

    for i in range(len(freq)):
        freq[i] = freq[i] / entries * 100

    x = intervals[:-1]
    x.append(50)

    plt.bar(x, freq, width=0.8)
    plt.xlabel("Percent error +-")
    plt.ylabel("Frequency %")
    plt.show()


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
