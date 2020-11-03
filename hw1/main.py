import numpy as np
from PLA_function import PLA_function

N = 1000
updates_list = []
w_0_list = []

for i in range(N):
    seed = i
    updates, w_0 = PLA_function(seed)
    updates_list.append(updates)
    w_0_list.append(w_0)

print(np.median(updates_list))
print(np.median(w_0_list))