from Config import Config
from DataHandler import DataHandler
from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
import numpy as np


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
        for i in range(24):
            cfg = self.cfgs[i]
            model = self.models[i]
            dh = self.dhs[i]

            X = np.array(dh.train_input)
            y = np.array(dh.train_labels)

            model.fit(X, y, epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=2)

    def eval_models(self, metrics):
        evals = []

        for i in range(24):
            cfg = self.cfgs[i]
            model = self.models[i]
            dh = self.dhs[i]

            X = np.array(dh.eval_input)
            y = np.array(dh.eval_labels)

            model.fit(X, y, epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=2)

            evaluator = ModelEvaluator(cfg, model, dh)

            evals.append(evaluator.evaluate(metrics))

        metrics_combined = [0 for m in range(len(metrics))]

        for eval_value in evals:
            for m in range(len(metrics)):
                metrics_combined[m] += eval_value[m]

        for i in range(len(metrics)):
            metrics_combined[i] = metrics_combined[i] / len(evals)

        return evals, metrics_combined


