from MMF import MMF
from SMF import SMF
import EvalMetrics

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

eval_indi, evals = smf.eval_model([EvalMetrics.mape])

print("SMF")
print("Indivdual model evals")
for e in eval_indi:
    print(e)

print("Total")
print(evals)

mmf.plot_days()
smf.plot_days()


