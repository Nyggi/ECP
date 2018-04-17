from ModelBuilder import ModelBuilder
from Tools import *
from Config import *
from Evaluator import Evaluator
from DataHandler import DataHandler

cfg = SingleConfig()

INPUT_SHAPE = (cfg.DAYS * 24,)

dh = DataHandler(cfg, 5)

mb = ModelBuilder(cfg, INPUT_SHAPE)

model = mb.nn()

print("Fitting model")
fitted_model = fit_model(cfg, model, dh.train_input, dh.train_labels)

print("------------------Evaluation-------------------")
evaluation = fitted_model.evaluate(np.array(dh.eval_input), np.array(dh.eval_labels), cfg.BATCH_SIZE, verbose=0)

for thing in evaluation:
    print(thing)

evaluator = Evaluator(model, dh.eval_input, dh.eval_labels)

evaluator.evaluate()

evaluator.evaluate_freq()

evaluator.weight_mmma()


