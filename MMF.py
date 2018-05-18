from Config import Config
from DataHandler import DataHandler
from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
import numpy as np
import matplotlib.pyplot as plt
import EvalMetrics
from keras import callbacks


class MMF:
    def __init__(self, house_id=None, verbose=False):
        self.verbose = verbose
        self.cfgs = self.create_configs(house_id)
        self.dhs = self.create_datahandlers()
        self.models = self.create_models()

    def create_configs(self, house_id):
        configs = []
        if self.verbose:
            print('CREATING CONFIGS')

        for i in range(24):
            cfg = Config()

            if house_id:
                cfg.HOUSE_ID = house_id

            cfg.HOUR_TO_PREDICT = i

            configs.append(cfg)

        return configs

    def create_datahandlers(self):
        dhs = []
        if self.verbose:
            print('CREATING DATAHANDLERS')

        for c in self.cfgs:
            if self.verbose:
                print(c.HOUR_TO_PREDICT)
            dhs.append(DataHandler(c))

        return dhs

    def create_models(self):
        models = []
        if self.verbose:
            print('CREATING MODELS')

        for i in range(24):
            input_shape = (len(self.dhs[i].train_input[0]),)
            mb = ModelBuilder(self.cfgs[i], input_shape)

            models.append(mb.nn_w())

        return models

    def train_models(self):
        if self.verbose:
            print("Training models")
        for i in range(24):
            if self.verbose:
                print(i)
            cfg = self.cfgs[i]
            model = self.models[i]
            dh = self.dhs[i]

            train_input = dh.train_input + dh.eval_input
            train_labels = dh.train_labels + dh.eval_labels

            X = np.array(train_input)
            y = np.array(train_labels)

            es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min')

            model.fit(X, y, epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0, callbacks=[es], validation_split=1 - cfg.TRAINING_CUT)

    def eval_models(self, metrics):
        evals = []

        for i in range(24):
            cfg = self.cfgs[i]
            model = self.models[i]
            dh = self.dhs[i]

            evaluator = ModelEvaluator(cfg, model, dh)

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
            cfg = self.cfgs[i]
            model = self.models[i]
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

        best_predictions = DataHandler.consumption_to_kWh(day_eval[0][1])
        best_targets = DataHandler.consumption_to_kWh(day_eval[0][2])
        line_up, = plt.plot(best_predictions, label='Prediction')
        line_down, = plt.plot(best_targets, label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.xlim(xmin=0, xmax=23)
        plt.minorticks_on()
        plt.subplots_adjust(top=1)
        plt.xlabel('Hour of the day')
        plt.ylabel('Energy consumption (kWh)')
        plt.text(0.5, 0.4, f'MAPE: {day_eval[0][0]:.1f} %')
        plt.figure()

        print(f'Mape: {day_eval[-1][0]:.2f} Day: {day_eval[-1][3]} Month: {day_eval[-1][4]}')

        worst_predictions = DataHandler.consumption_to_kWh(day_eval[-1][1])
        worst_targets = DataHandler.consumption_to_kWh(day_eval[-1][2])
        line_up, = plt.plot(worst_predictions, label='Prediction')
        line_down, = plt.plot(worst_targets, label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.xlim(xmin=0, xmax=23)
        plt.minorticks_on()
        plt.subplots_adjust(top=1)
        plt.xlabel('Hour of the day')
        plt.ylabel('Energy consumption (kWh)')
        plt.text(0.5, 0.4, f'MAPE: {day_eval[-1][0]:.1f} %')
        plt.figure()

        errors = []

        for e in day_eval:
            errors.append(e[0])

        ModelEvaluator.plot_abs_error_freq(errors)
        ModelEvaluator.plot_cumu_abs_error_freq(errors)





