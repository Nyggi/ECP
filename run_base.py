from BaselineModel import BaselineModel
import numpy as np

bl = BaselineModel(5)

result = bl.eval_model()

for hour in result:
    print(hour)

print(np.mean(result, axis=0))