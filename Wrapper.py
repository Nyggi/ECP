import numpy as np
import sys
from Config import Config
from sklearn.ensemble import RandomForestRegressor
from DataHandler import DataHandler
from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
import EvalMetrics


class Wrapper:
    def __init__(self, eval_metric):
        self.eval_metric = eval_metric

    def best_first_ff(self):
        cfg = Config()
        best_features = [0 for f in cfg.FEATURES]

        best_eval = sys.maxsize

        while True:
            evals = []

            for i in range(len(best_features)):

                current_features = best_features.copy()
                current_features[i] = 1

                cfg.FEATURES = current_features

                eval_value = self.eval_subset(cfg)

                evals.append(eval_value)

            new_best_eval = min(evals)
            index = evals.index(new_best_eval)

            if new_best_eval < best_eval and not best_features[index]:
                best_features[index] = 1
                best_eval = new_best_eval
            else:
                break

        return best_features, best_eval

    def best_first_bw(self):
        cfg = Config()
        best_features = [1 for f in cfg.FEATURES]

        best_eval = sys.maxsize

        while True:
            evals = []

            for i in range(len(best_features)):

                current_features = best_features.copy()
                current_features[i] = 0

                cfg.FEATURES = current_features

                eval_value = self.eval_subset(cfg)

                evals.append(eval_value)

            new_best_eval = min(evals)
            index = evals.index(new_best_eval)

            if new_best_eval < best_eval and best_features[index]:
                best_features[index] = 0
                best_eval = new_best_eval
            else:
                break

        return best_features, best_eval

    def eval_subset(self, cfg):
        dh = DataHandler(cfg)

        # model = RandomForestRegressor()
        model = ModelBuilder(cfg, (len(dh.train_input[0]),)).nn_w()
        X = np.array(dh.train_input)
        y = np.array(dh.train_labels)

        # model.fit(X, y)
        model.fit(X, y, epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)

        evaluator = ModelEvaluator(cfg, model, dh)

        eval_value = evaluator.evaluate([EvalMetrics.mape])[0]

        return eval_value




