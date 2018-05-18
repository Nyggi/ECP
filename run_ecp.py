from MMF import MMF
from SMF import SMF
import EvalMetrics
import matplotlib.pyplot as plt
from ModelEvaluator import ModelEvaluator


mmf = MMF()
smf = SMF()

mmf.train_models()
smf.train_model()

eval_indi, evals = mmf.eval_models([EvalMetrics.mape])
print("MMF")
print("Indivdual model evals")
for e in eval_indi:
    print(e)

print("Total")
print(evals)

ModelEvaluator.plot_mape_on_day(eval_indi, 'MMF')

eval_indi, evals = smf.eval_model([EvalMetrics.mape])

print("SMF")
print("Indivdual model evals")
for e in eval_indi:
    print(e)

print("Total")
print(evals)

ModelEvaluator.plot_mape_on_day(eval_indi, 'SMF')

print("MMF")
mmf.plot_days()
print("SMF")
smf.plot_days()
mmf.plot_residual()
smf.plot_residual()

plt.show()
