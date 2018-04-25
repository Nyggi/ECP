from ModelBuilder import ModelBuilder
from Tools import *
from Config import *
from ModelEvaluator import ModelEvaluator
from DataHandler import DataHandler

cfg = SingleConfig()

dh = DataHandler(cfg, 5)

INPUT_SHAPE = (len(dh.train_input[0]),)

mb = ModelBuilder(cfg, INPUT_SHAPE)

model = mb.nn_small()

print("Fitting model")
model.fit(np.array(dh.train_input), np.array(dh.train_labels), epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=2)

print("------------------Evaluation-------------------")
evaluator = ModelEvaluator(model, dh.eval_input, dh.eval_labels)

evaluator.evaluate(dh.scaler)

evaluator.evaluate_freq(dh.scaler)

evaluator.weight_mmma()

evaluator.weight_mmma_plot()

plt.show()
