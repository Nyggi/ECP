import numpy as np
import sys
from keras.models import clone_model
import matplotlib.pyplot as plt
from ModelEvaluator import ModelEvaluator
import EvalMetrics


def fit_model(cfg, model, dh, show_error_graph=False):
    best_mape = sys.maxsize

    best_model = clone_model(model)
    best_model.compile(loss=cfg.LOSS, optimizer=cfg.OPTIMIZER, metrics=["mape", "mae"])

    mape_list_train = list()
    mape_list_eval = list()

    evaluator = ModelEvaluator(cfg, model, dh)

    metrics = [EvalMetrics.mape]

    for e in range(cfg.EPOCHS):
        model.fit(np.array(dh.train_input), np.array(dh.train_labels), cfg.BATCH_SIZE, 1, verbose=0, shuffle=True)

        mape_train = evaluator.evaluate_data(metrics, dh.train_input, dh.train_labels)[0]
        mape_eval = evaluator.evaluate(metrics)[0]

        mape_list_train.append(mape_train)
        mape_list_eval.append(mape_eval)

        print("Epoch " + str(e + 1) + '/' + str(cfg.EPOCHS) + ' - MAPE: ' + str(mape_train))

        if mape_train < best_mape:
            best_mape = mape_train
            # Copy weigths from model to best model
            best_model.set_weights(model.get_weights())

    if show_error_graph is True:
        training_plot, = plt.plot(mape_list_train, label='Training', color='blue')
        eval_plot, = plt.plot(mape_list_eval, label='Evaluation', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('MAPE')
        plt.legend(handles=[training_plot, eval_plot])

        plt.show()

    return best_model


# A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names=None, sort=False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                      for coef, name in lst)
