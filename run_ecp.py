from MMF import MMF
from SMF import SMF
import EvalMetrics
import matplotlib.pyplot as plt
from ModelEvaluator import ModelEvaluator

MMF_R = False
SMF_R = True

if MMF_R:
    mmf = MMF()
    mmf.train_models()

    eval_indi, evals = mmf.eval_models([EvalMetrics.mape])
    print("MMF")
    print("Indivdual model evals")
    for e in eval_indi:
        print(e)
    print("Total")
    print(evals)

    ModelEvaluator.plot_mape_on_day(eval_indi, 'MMF')

if SMF_R:
    smf = SMF()
    smf.train_model()
    eval_indi, evals = smf.eval_model([EvalMetrics.mape])

    print("SMF")
    print("Indivdual model evals")
    for e in eval_indi:
        print(e)

    print("Total")
    print(evals)

    ModelEvaluator.plot_mape_on_day(eval_indi, 'SMF')

if MMF_R:
    print("MMF")
    mmf.plot_days()
    mmf.plot_residual()

if SMF_R:
    print("SMF")
    smf.plot_days()
    smf.plot_residual()

plt.show()
