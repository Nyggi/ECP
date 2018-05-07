from Wrapper import Wrapper
from DataHandler import DataHandler

w = Wrapper()

best_ff, eval_ff = w.best_first_ff()

print(best_ff)
print("Eval: " + str(eval_ff))

best_bw, eval_bw = w.best_first_bw()

print(best_bw)
print("Eval: " + str(eval_bw))
