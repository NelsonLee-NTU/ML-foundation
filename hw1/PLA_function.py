import numpy as np
import random

def PLA_function(seed):
    random.seed(seed)

    # read trained data
    x = np.genfromtxt('hw1_train.dat')

    # add x_0 = 1 into data
    N = x.shape[0]
    x = np.c_[0*np.ones(N), x]

    # initialize the w vector
    w = np.zeros((11, 1))
    w_T = w.transpose()

    iteration = 0
    updates = 0
    # main updating loop
    while iteration < 5*N:
        rand = random.randrange(N)
        x_picked = np.array([x[rand, 0:11]]).transpose() / 4
        y_picked = x[rand, 11]
        h = np.sign(w_T.dot(x_picked)) if np.sign(w_T.dot(x_picked)) != 0 else -1

        if h != y_picked:
            w += y_picked * x_picked
            w_T = w.transpose()
            iteration = 0
            updates += 1
        else:
            iteration += 1

    return updates, w[0]