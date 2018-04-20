from ModelBuilder import ModelBuilder
from Tools import *
from Config import *
from Evaluator import Evaluator
from DataHandler import DataHandler

cfg = SingleConfig()

dh = DataHandler(cfg, 5)

INPUT_SHAPE = (len(dh.train_input[0]),)

mb = ModelBuilder(cfg, INPUT_SHAPE)

model = mb.nn_small()

print("Fitting model")
fitted_model = fit_model(cfg, model, dh.train_input, dh.train_labels)

print("------------------Evaluation-------------------")
evaluator = Evaluator(fitted_model, dh.eval_input, dh.eval_labels)

evaluator.evaluate(dh.scaler)

evaluator.evaluate_freq(dh.scaler)

evaluator.weight_mmma()

evaluator.weight_mmma_plot()

plt.show()
