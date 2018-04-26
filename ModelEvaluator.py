import json
import matplotlib.pyplot as plt
import numpy as np


class ModelEvaluator:

    def __init__(self, model, eval_input, eval_labels):
        self.model = model
        self.eval_input = eval_input
        self.eval_labels = eval_labels

    def weight_mmma(self):
        weights = self.model.get_weights()

        layers = []

        for layer in weights:
            nodes = []
            for node in layer:
                weight_min = min(node)
                weight_max = max(node)
                median = sorted(node)[len(node) // 2]
                average = sum(node) / len(node)

                weight_string = f'MIN {weight_min:.4f}, MAX {weight_max:.4f}, Median {median:.4f}, AVG {average:.4f}'
                nodes.append(weight_string)

            layers.append(nodes)

        with open('weights_mmma.json', 'w') as outfile:
            json_string = json.dumps(layers)
            outfile.write(json_string)

    def weight_mmma_plot(self):
        weights = self.model.get_weights()

        for layer in weights:

            if isinstance(layer[0], np.ndarray):
                weight_min = []
                weight_max = []
                median = []
                average = []

                for node in layer:
                    weight_min.append(min(node))
                    weight_max.append(max(node))
                    median.append(sorted(node)[len(node) // 2])
                    average.append(sum(node) / len(node))

                line_min, = plt.plot(weight_min, label='Minimum', color='blue')
                line_max, = plt.plot(weight_max, label='Maximum', color='orange')
                line_median, = plt.plot(median, label='Median', color='green')
                line_average, = plt.plot(average, label='Average', color='red')
                plt.legend(handles=[line_min, line_max, line_median, line_average])

                x = np.arange(len(layer))
                z = np.polyfit(x, weight_min, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), linestyle='dashed', color='blue')
                z = np.polyfit(x, weight_max, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), linestyle='dashed', color='orange')
                z = np.polyfit(x, median, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), linestyle='dashed', color='green')
                z = np.polyfit(x, average, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), linestyle='dashed', color='red')
                plt.figure()

    def plot_cumu_abs_error_freq(self, errors):
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

    def plot_error_freq(self, errors):
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

    def evaluate_freq(self, scaler=None):
        i = 0
        errors = []

        for inputD in self.eval_input:
            res, label = self._predict_and_shape(inputD, self.eval_labels[i], scaler)

            error = (res - label) / label * 100
            errors.extend(error)
            i += 1

        self.plot_error_freq(errors)
        self.plot_cumu_abs_error_freq(errors)

    def evaluate(self, scaler=None):
        i = 0
        total_error = 0
        total_error_percent = 0

        max_error = 0

        predictions = []
        labels = []
        for inputD in self.eval_input:
            res, label = self._predict_and_shape(inputD, self.eval_labels[i], scaler)

            predictions.append(res)
            labels.append(label)

            error = abs(res - label)

            mean_error = np.mean(error)

            if mean_error > max_error:
                max_error = mean_error

            total_error_percent += np.mean(error / label) * 100
            total_error += mean_error

            i += 1

        print("MAE:     " + f'{(total_error / i):.2f}')
        print("MAPE:    " + f'{(total_error_percent / i):.2f}')
        print("MAX:     " + f'{max_error:.2f}')

        line_up, = plt.plot(predictions, label='Prediction')
        line_down, = plt.plot(labels, label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.figure()

    def _predict_and_shape(self, inputD, label, scaler=None):
        predict = np.array([inputD])
        p_1 = self.model.predict(predict)

        res = p_1
        label = np.asarray(label)

        if scaler is not None:
            res = scaler.inverse_transform(res)
            label = scaler.inverse_transform(label.reshape(1, -1))

        res = np.reshape(res, -1)
        label = np.reshape(label, -1)

        return res, label
