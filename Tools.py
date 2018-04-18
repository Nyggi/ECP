import numpy as np
from keras.models import clone_model


def fit_model(cfg, model, train_input, train_labels):
    best_mape = 999999999

    best_model = clone_model(model)
    best_model.compile(loss=cfg.LOSS, optimizer=cfg.OPTIMIZER, metrics=["mape"])

    for e in range(cfg.EPOCHS):
        stats = model.fit(np.array(train_input), np.array(train_labels), cfg.BATCH_SIZE, 1, verbose=0, shuffle=True)
        mape = stats.history['mean_absolute_percentage_error'][0]

        print("Epoch " + str(e) + '/' + str(cfg.EPOCHS) + ' - MAPE: ' + str(mape))

        if mape < best_mape:
            best_mape = mape
            # Copy weigths from model to best model
            best_model.set_weights(model.get_weights())

    return best_model

