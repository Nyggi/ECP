from Config import Config
from DataHandler import DataHandler
from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
import numpy as np
import matplotlib.pyplot as plt
import EvalMetrics


class ECP:
    def __init__(self, house_id=None):
        self.cfgs = self.create_configs(house_id)
        self.dhs = self.create_datahandlers()
        self.models = self.create_models()

    def create_configs(self, house_id):
        configs = []

        for i in range(24):
            cfg = Config()

            if house_id:
                cfg.HOUSE_ID = house_id

            cfg.HOUR_TO_PREDICT = i

            configs.append(cfg)

        return configs

    def create_datahandlers(self):
        dhs = []

        for c in self.cfgs:
            dhs.append(DataHandler(c))

        return dhs

    def create_models(self):
        models = []

        for i in range(24):
            input_shape = (len(self.dhs[i].train_input[0]),)
            mb = ModelBuilder(self.cfgs[i], input_shape)

            models.append(mb.nn_w())

        return models

    def train_models(self):
        print("Training models")
        for i in range(24):
            print(i)
            cfg = self.cfgs[i]
            model = self.models[i]
            dh = self.dhs[i]

            X = np.array(dh.train_input)
            y = np.array(dh.train_labels)

            model.fit(X, y, epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)

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

        line_up, = plt.plot(day_eval[0][1], label='Prediction')
        line_down, = plt.plot(day_eval[0][2], label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.figure()

        print(f'Mape: {day_eval[-1][0]:.2f} Day: {day_eval[-1][3]} Month: {day_eval[-1][4]}')

        line_up, = plt.plot(day_eval[-1][1], label='Prediction')
        line_down, = plt.plot(day_eval[-1][2], label='Target')
        plt.legend(handles=[line_up, line_down])
        plt.figure()

        plt.show()




