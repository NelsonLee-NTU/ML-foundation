import numpy as np
import random

# read train data
x = np.genfromtxt('hw1_train.dat')

# add x_0 = 1 into data
x = np.c_[np.ones(100), x]
N = x.shape[0]

# initialize the w vector
w = np.zeros((11, 1))
w_T = w.transpose()


iteration = 0
runtime = 0
while iteration < 5*N:
    # print("iteration = ", iteration)
    rand = random.randrange(N)
    x_picked = np.array([x[rand, 0:11]]).transpose()
    # print("x_picked = ", x_picked)
    y_picked = x[rand, 11]
    # print("y_picked = ", y_picked)

    # print("w_T = ", w_T)
    # print("h_before = ", w_T.dot(x_picked))

    h = np.sign(w_T.dot(x_picked)) if np.sign(w_T.dot(x_picked)) != 0 else -1
    # print("h = ", h)
    
    if h != y_picked:
        # print("no")
        # print("y_picked * x_picked = ", y_picked * x_picked)
        w += y_picked * x_picked
        # print("w = ", w)
        w_T = w.transpose()
        iteration = 0
        runtime += 1
    else:
        # print("yes")
        iteration += 1

print("runtime = ", runtime)
print("w = ", w)

