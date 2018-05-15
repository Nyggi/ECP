from MMF import MMF
from SMF import SMF
import EvalMetrics
import matplotlib.pyplot as plt
import numpy as np

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



mapes = np.array(eval_indi).reshape(-1)
plot, = plt.plot(mapes, label='MAPE', color='red')
plt.axhline(np.mean(mapes), linestyle='dashed', color='red')
plt.legend(handles=[plot])
plt.xlim(xmin=0, xmax=23)
plt.minorticks_on()
plt.subplots_adjust(top=1)
plt.xlabel('Hour of the day')
plt.ylabel('MAPE %')
plt.text(0.5, 20, f'MAPE: {np.mean(mapes):.1f} %')
plt.show()