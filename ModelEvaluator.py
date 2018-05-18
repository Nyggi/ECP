import matplotlib.pyplot as plt
import numpy as np
import EvalMetrics


class ModelEvaluator:

    def __init__(self, cfg, model, dh):
        self.cfg = cfg
        self.model = model
        self.dh = dh

    def scale_values(self, predictions, labels):
        predictions = self.dh.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        labels = self.dh.scaler.inverse_transform(np.array(labels).reshape(-1, 1))

        return predictions, labels

    def get_eval_data(self):
        X_eval = np.array(self.dh.eval_input)
        y_eval = np.array(self.dh.eval_labels)

        predictions = self.model.predict(X_eval)

        if self.cfg.SCALE_VALUES:
            predictions, y_eval = self.scale_values(predictions, y_eval)

        return predictions, y_eval

    def get_evaluation_per_metric(self, metrics, predictions, labels):
        eval_values = []

        for metric in metrics:
            eval_values.append(metric(predictions, labels))

        return eval_values

    def evaluate_freq(self):
        predictions, y_eval = self.get_eval_data()

        eval_value = EvalMetrics.pe(predictions, y_eval)

        self.plot_error_freq(eval_value)
        self.plot_cumu_abs_error_freq(eval_value)

    def evaluate(self, metrics):
        predictions, y_eval = self.get_eval_data()

        return self.get_evaluation_per_metric(metrics, predictions, y_eval)

    def evaluate_data(self, metrics, inputs, labels):
        X_eval = np.array(inputs)
        y_eval = np.array(labels)

        predictions = self.model.predict(X_eval)

        if self.cfg.SCALE_VALUES:
            predictions, y_eval = self.scale_values(predictions, y_eval)

        return self.get_evaluation_per_metric(metrics, predictions, y_eval)

    def plot_prediction(self):
        predictions, y_eval = self.get_eval_data()

        line_up, = plt.plot(predictions, label='Prediction')
        line_down, = plt.plot(y_eval, label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.figure()

    def plot_weight_mmma(self):
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def plot_abs_error_freq(errors):
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

        plt.hist(abs_errors, bins=bins_list, weights=weights, edgecolor='black', linewidth=0.5)
        plt.axvline(np.mean(abs_errors), color='k', linestyle='dashed', linewidth=1)
        plt.xlim(xmin=0, xmax=bin_max)
        plt.xlabel("MAPE %")
        plt.ylabel("Frequency %")
        plt.xticks(np.arange(0, bin_max + 1, step=10))
        plt.minorticks_on()
        plt.subplots_adjust(top=1)
        plt.figure()

    @staticmethod
    def plot_residual(all_predictions, all_y_eval, title):
        z = np.polyfit(all_predictions, all_y_eval, 1)
        p = np.poly1d(z)
        difference = all_y_eval - p(all_predictions)
        # std = np.std(difference, ddof=1)
        # standardized_residual = difference / std
        plt.plot(all_predictions, difference, '.', alpha=0.1, color='black')
        plt.axhline(0, color='grey', alpha=0.75)
        plt.xlabel("Predicted consumption")
        plt.ylabel("Residual")
        plt.title(title)
        plt.minorticks_on()
        plt.figure()

    @staticmethod
    def plot_mape_on_day(mape_list, title):
        mapes = np.array(mape_list).reshape(-1)
        plot, = plt.plot(mapes, label='MAPE', color='red')
        plt.axhline(np.mean(mapes), linestyle='dashed', color='red')
        plt.legend(handles=[plot])
        plt.xlim(xmin=0, xmax=23)
        plt.minorticks_on()
        plt.subplots_adjust(top=1)
        plt.xlabel('Hour of the day')
        plt.ylabel('MAPE %')
        plt.text(0.5, 20, f'MAPE: {np.mean(mapes):.1f} %')
        plt.title(title)
        plt.figure()

    @staticmethod
    def plot_day_prediction(predictions, targets, mape):
        line_up, = plt.plot(predictions, label='Prediction')
        line_down, = plt.plot(targets, label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.xlim(xmin=0, xmax=23)
        plt.minorticks_on()
        plt.subplots_adjust(top=1)
        plt.xlabel('Hour of the day')
        plt.ylabel('Energy consumption (kWh)')
        plt.text(0.5, 0.4, f'MAPE: {mape:.1f} %')
        plt.figure()

    @staticmethod
    def plot_week_prediction(predictions, targets, mape):
        line_up, = plt.plot(predictions, label='Prediction')
        line_down, = plt.plot(targets, label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.xlim(xmin=0, xmax=167)
        plt.minorticks_on()
        plt.subplots_adjust(top=1)
        plt.xlabel('Hour of the day')
        plt.ylabel('Energy consumption (kWh)')
        plt.text(0.5, 0.8, f'MAPE: {mape:.1f} %')
        plt.figure()
