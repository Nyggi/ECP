from ModelBuilder import ModelBuilder
from Tools import *
from Config import *

AGGREGATION_LEVEL = 'hourly'
# lighting, fridge, households, electronics, inductive
GROUP = "households"

cfg = SingleConfig()

INPUT_SHAPE = (cfg.DAYS * 24,)

data = fetch_data(cfg.AMOUNTOFHOUSES)

train_input, train_labels, eval_input, eval_labels = construct_training_data(cfg, data)

mb = ModelBuilder(cfg, INPUT_SHAPE)

model = mb.nn()

print("Fitting model")
fitted_model = fit_model(cfg, model, train_input, train_labels)

print("------------------Evaluation-------------------")
evaluation = fitted_model.evaluate(np.array(eval_input), np.array(eval_labels), cfg.BATCH_SIZE, verbose=0)

for thing in evaluation:
    print(thing)

evaluate(fitted_model, eval_input, eval_labels, cfg.GRAPH_CUT)

make_correlation_matrix()

# evaluate_freq(best_model, eval_input, eval_labels)

# evaluate_other(cfg, fitted_model, AGGREGATION_LEVEL, GROUP, 1)
