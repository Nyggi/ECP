from Wrapper import Wrapper
import EvalMetrics

w = Wrapper(EvalMetrics.mape)

best_ff, eval_ff = w.best_first_ff()

print(best_ff)
print("Eval: " + str(eval_ff))

best_bw, eval_bw = w.best_first_bw()

print(best_bw)
print("Eval: " + str(eval_bw))
