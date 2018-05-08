from ModelBuilder import ModelBuilder
from Tools import *
from Config import Config
from ModelEvaluator import ModelEvaluator
from DataHandler import DataHandler
import EvalMetrics


cfg = Config()
dh = DataHandler(cfg)

INPUT_SHAPE = (len(dh.train_input[0]),)

mb = ModelBuilder(cfg, INPUT_SHAPE)
model = mb.nn_w()

model.fit(np.array(dh.train_input), np.array(dh.train_labels), epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=2)

evaluator = ModelEvaluator(cfg, model, dh)

eval_values = evaluator.evaluate([EvalMetrics.mape, EvalMetrics.mer, EvalMetrics.mse])

print("Evaluations")
for eval_value in eval_values:
    print(f'{eval_value:.2f}')

evaluator.plot_prediction()

evaluator.evaluate_freq()

evaluator.plot_weight_mmma()

plt.show()
