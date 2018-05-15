from Config import Config
from DataHandler import DataHandler
from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
import numpy as np
import matplotlib.pyplot as plt
import EvalMetrics


class SMF:
    def __init__(self, cfg=None, house_id=None, verbose=False):
        self.verbose = verbose
        if not cfg:
            self.cfg = self.create_config(house_id)
        else:
            self.cfg = cfg
        self.dhs = self.create_datahandlers()
        self.model = self.create_models()

    def create_config(self, house_id):
        if self.verbose:
            print('CREATING CONFIG')
        cfg = Config()

        if house_id:
            cfg.HOUSE_ID = house_id

        return cfg

    def create_datahandlers(self):
        dhs = []
        if self.verbose:
            print('CREATING DATAHANDLERS')

        for h in range(24):
            if self.verbose:
                print(h)
            cfg = Config()
            cfg.HOUSE_ID = self.cfg.HOUSE_ID
            cfg.HOUR_TO_PREDICT = h
            cfg.FEATURES = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
            # [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0] FF
            # [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1] BW
            dhs.append(DataHandler(cfg))

        return dhs

    def create_models(self):
        if self.verbose:
            print('CREATING MODEL')

        input_shape = (len(self.dhs[0].train_input[0]),)
        mb = ModelBuilder(self.cfg, input_shape)

        model = mb.nn_w()

        return model

    def train_model(self):
        if self.verbose:
            print("Training model")
        train_input = []
        train_labels = []

        for i in range(24):
            dh = self.dhs[i]
            train_input.extend(dh.train_input)
            train_labels.extend(dh.train_labels)

        X = np.array(train_input)
        y = np.array(train_labels)

        self.model.fit(X, y, epochs=self.cfg.EPOCHS, batch_size=self.cfg.BATCH_SIZE, verbose=0)

    def eval_model(self, metrics):
        evals = []

        for i in range(24):
            dh = self.dhs[i]

            evaluator = ModelEvaluator(self.cfg, self.model, dh)

            evals.append(evaluator.evaluate(metrics))

        metrics_combined = [0 for m in range(len(metrics))]

        for eval_value in evals:
            for m in range(len(metrics)):
                metrics_combined[m] += eval_value[m]

        for i in range(len(metrics)):
            metrics_combined[i] = metrics_combined[i] / len(evals)

        return evals, metrics_combined

    def plot_days(self):
        predictions_data = []
        y_evals = []

        for i in range(24):
            cfg = self.cfg
            model = self.model
            dh = self.dhs[i]

            evaluator = ModelEvaluator(cfg, model, dh)

            predictions, y_eval = evaluator.get_eval_data()

            predictions_data.append(predictions)
            y_evals.append(y_eval)

        predictions_day = []
        y_eval_day = []

        # -1 because hour 23 have one less entry than the other hours
        for i in range(len(predictions_data[0]) - 1):
            day_p = []
            day_e = []

            for h in range(24):
                day_p.append(predictions_data[h][i])
                day_e.append(y_evals[h][i])

            predictions_day.append(day_p)
            y_eval_day.append(day_e)

        day_eval = []

        for i in range(len(predictions_day)):
            p = predictions_day[i]
            e = y_eval_day[i]
            mape = EvalMetrics.mape(np.array(p), np.array(e))

            day_eval.append([mape, p, e, i % 7, i // 30])

        day_eval.sort(key=lambda x: x[0])

        print(f'Mape: {day_eval[0][0]:.2f} Day: {day_eval[0][3]} Month: {day_eval[0][4]}')

        line_up, = plt.plot(day_eval[0][1], label='Prediction')
        line_down, = plt.plot(day_eval[0][2], label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.figure()

        print(f'Mape: {day_eval[-1][0]:.2f} Day: {day_eval[-1][3]} Month: {day_eval[-1][4]}')

        line_up, = plt.plot(day_eval[-1][1], label='Prediction')
        line_down, = plt.plot(day_eval[-1][2], label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.figure()

        errors = []

        for e in day_eval:
            errors.append(e[0])

        ModelEvaluator.plot_error_freq(errors)

        plt.show()
