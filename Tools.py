import numpy as np
import sys
from keras.models import clone_model
import matplotlib.pyplot as plt


def fit_model(cfg, model, train_input, train_labels, show_error_graph=False):
    best_mape = sys.maxsize

    best_model = clone_model(model)
    best_model.compile(loss=cfg.LOSS, optimizer=cfg.OPTIMIZER, metrics=["mape", "mae"])

    mape_list = list()
    mae_list = list()

    for e in range(cfg.EPOCHS):
        stats = model.fit(np.array(train_input), np.array(train_labels), cfg.BATCH_SIZE, 1, verbose=0, shuffle=True)
        mape = stats.history['mean_absolute_percentage_error'][0]
        mae = stats.history['mean_absolute_error'][0]

        mape_list.append(mape)
        mae_list.append(mae)

        print("Epoch " + str(e + 1) + '/' + str(cfg.EPOCHS) + ' - MAPE: ' + str(mape))

        if mape < best_mape:
            best_mape = mape
            # Copy weigths from model to best model
            best_model.set_weights(model.get_weights())

    if show_error_graph is True:
        fig, ax1 = plt.subplots()
        mape_plot = ax1.plot(mape_list, label='MAPE', color='blue')

        ax2 = ax1.twinx()
        mae_plot = ax2.plot(mae_list, label='MAE', color='red')

        lns = mape_plot + mae_plot
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('MAPE')
        ax2.set_ylabel('MAE')

        fig.tight_layout()
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
