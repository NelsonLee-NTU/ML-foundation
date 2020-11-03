import numpy as np
import random
from Decision_Stump import decision_stump

size = 200
ans = []
tau = 0.1

for i in range(10000):
    if i % 100 == 0:
        print(i)
    x = np.random.uniform(-1, 1, size)
    x.sort()
    # print(x)

    ans.append(decision_stump(x, size, tau))

print(np.mean(ans))