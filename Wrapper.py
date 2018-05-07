import numpy as np
import sys
from Config import SingleConfig
from sklearn.ensemble import RandomForestRegressor
from DataHandler import DataHandler


class Wrapper:
    def __init__(self):
        pass

    def best_first_ff(self):
        cfg = SingleConfig()
        best_features = [0 for f in cfg.FEATURES]

        best_eval = sys.maxsize

        while True:
            evals = []

            for i in range(len(best_features)):

                current_features = best_features.copy()
                current_features[i] = 1

                cfg.FEATURES = current_features

                eval_metric = self.eval_subset(cfg)

                evals.append(eval_metric)

            new_best_eval = min(evals)
            index = evals.index(new_best_eval)

            if new_best_eval < best_eval:
                best_features[index] = 1
                best_eval = new_best_eval
            else:
                break

        return best_features, best_eval

    def best_first_bw(self):
        cfg = SingleConfig()
        best_features = [1 for f in cfg.FEATURES]

        best_eval = sys.maxsize

        while True:
            evals = []

            for i in range(len(best_features)):

                current_features = best_features.copy()
                current_features[i] = 0

                cfg.FEATURES = current_features

                eval_metric = self.eval_subset(cfg)

                evals.append(eval_metric)

            new_best_eval = min(evals)
            index = evals.index(new_best_eval)

            if new_best_eval < best_eval:
                best_features[index] = 0
                best_eval = new_best_eval
            else:
                break

        return best_features, best_eval

    def eval_subset(self, cfg):
        dh = DataHandler(cfg)

        rf = RandomForestRegressor()
        X = np.array(dh.train_input)
        y = np.array(dh.train_labels)

        rf.fit(X, y)

        X_eval = np.array(dh.eval_input)
        y_eval = np.array(dh.eval_labels)

        predictions = rf.predict(X_eval)

        predictions = dh.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        y_eval = dh.scaler.inverse_transform(np.array(y_eval).reshape(-1, 1))

        #eval_metric = self.mape(predictions, y_eval)
        eval_metric = self.mse(predictions, y_eval)

        return eval_metric

    def mape(self, predictions, y_eval):
        errors = abs(predictions - y_eval)
        ape = errors / abs(y_eval) * 100
        mape = sum(ape) / len(y_eval)

        return mape

    def mse(self, predictions, y_eval):
        errors = (predictions - y_eval)**2
        mse = sum(errors) / len(y_eval)

        return mse




