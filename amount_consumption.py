from MMF import MMF
from SMF import SMF
import EvalMetrics
import matplotlib.pyplot as plt
from Config import Config
import numpy as np
from ModelEvaluator import ModelEvaluator
from DataHandler import DataHandler


def eval_model_smf(smf, metrics, base_dhs):
    evals = []

    for i in range(24):
        dh = smf.dhs[i]

        evaluator = ModelEvaluator(smf.cfg, smf.model, dh)

        base_dh = base_dhs[i]
        evals.append(evaluator.evaluate_data(metrics, base_dh.eval_input, base_dh.eval_labels))

    metrics_combined = [0 for m in range(len(metrics))]

    for eval_value in evals:
        for m in range(len(metrics)):
            metrics_combined[m] += eval_value[m]

    for i in range(len(metrics)):
        metrics_combined[i] = metrics_combined[i] / len(evals)

    return metrics_combined


def create_datahandlers_smf():
    dhs = []

    for h in range(24):
        cfg = Config()
        cfg.SMF_FEATURES = True
        cfg.HOUR_TO_PREDICT = h
        dhs.append(DataHandler(cfg))

    return dhs


def run_smf(data_steps, reps):
    cfg = Config()
    base_dhs = create_datahandlers_smf()
    mapes = []

    for slice in data_steps:
        print("Starting slice " + str(slice))
        cfg.DATA_SLICE = slice
        smf = SMF(cfg=cfg)
        inter_mape = []
        for i in range(reps):
            print("Starting rep " + str(i))
            smf.model = smf.create_models()
            smf.train_model()
            evals = eval_model_smf(smf, [EvalMetrics.mape], base_dhs)
            inter_mape.append(evals[0])
        mapes.append(np.mean(inter_mape))

    return mapes


def create_configs_mmf():
    configs = []

    for i in range(24):
        cfg = Config()
        cfg.HOUR_TO_PREDICT = i
        configs.append(cfg)

    return configs


def create_datahandlers_mmf(cfgs):
    dhs = []

    for c in cfgs:
        dhs.append(DataHandler(c))

    return dhs


def eval_model_mmf(mmf, metrics, base_dhs):
    evals = []

    for i in range(24):
        cfg = mmf.cfgs[i]
        model = mmf.models[i]
        dh = mmf.dhs[i]

        evaluator = ModelEvaluator(cfg, model, dh)

        base_dh = base_dhs[i]
        evals.append(evaluator.evaluate_data(metrics, base_dh.eval_input, base_dh.eval_labels))

    metrics_combined = [0 for m in range(len(metrics))]

    for eval_value in evals:
        for m in range(len(metrics)):
            metrics_combined[m] += eval_value[m]

    for i in range(len(metrics)):
        metrics_combined[i] = metrics_combined[i] / len(evals)

    return metrics_combined


def run_mmf(data_steps, reps):
    cfgs = create_configs_mmf()
    base_dhs = create_datahandlers_mmf(cfgs)
    mapes = []

    for slice in data_steps:
        print("Starting slice " + str(slice))
        for cfg in cfgs:
            cfg.DATA_SLICE = slice

        mmf = MMF(cfgs=cfgs)
        inter_mape = []
        for i in range(reps):
            print("Starting rep " + str(i))
            mmf.models = mmf.create_models()
            mmf.train_models()
            evals = eval_model_mmf(mmf, [EvalMetrics.mape], base_dhs)
            inter_mape.append(evals[0])
        mapes.append(np.mean(inter_mape))

    return mapes


steps_week = [1, 2, 4, 8, 16, 32, 48, 72]
data_steps = np.array(steps_week) / 72

reps = 5
# print('SMF')
# mapes_smf = run_smf(data_steps, reps)
# print(mapes_smf)

steps_week = [72]
data_steps = np.array(steps_week) / 72

print('MMF')
mapes_mmf = run_mmf(data_steps, reps)
print(mapes_mmf)

# plot_smf, = plt.plot(steps_week, mapes_smf, label='MAPE')
# plot_mmf, = plt.plot(steps_week, mapes_mmf, label='MAPE')
# plt.legend(handles=[plot_smf, plot_mmf])
# plt.minorticks_on()
# plt.xlabel('Number of weeks')
# plt.ylabel('MAPE %')
# plt.show()
