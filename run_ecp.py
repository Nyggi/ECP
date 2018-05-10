from ECP import ECP
import EvalMetrics

ecp = ECP()

ecp.train_models()

eval_indi, evals = ecp.eval_models([EvalMetrics.mape])

print("Indivdual model evals")
for e in eval_indi:
    print(e)

print("Total")
print(evals)

ecp.plot_days()
