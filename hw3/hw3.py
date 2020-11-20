import numpy as np
import random
from numpy.linalg import inv
import math

# read train data
data = np.genfromtxt('hw3_train.dat')
data_test = np.genfromtxt('hw3_test.dat')
# x_test = np.genfromtxt('hw3_test.dat')

N = data.shape[0]
N_test = data_test.shape[0]
# print('N = ', N)
# print(data.shape)

# add x_0 = 1 into data
data = np.c_[np.ones(data.shape[0]), data]
data_test = np.c_[np.ones(data_test.shape[0]), data_test]
# print('shape of data = ', data.shape)

y = data[:, -1:]
y_test = data_test[:, -1:]
# print(type(y))
# print('shape of y = ', y.shape)


X = data[:, :-1]
X_test = data_test[:, :-1]
# print('shape of X = ', X.shape)

# X_dagger = np.linalg.inv(X.T.dot(X)).dot(X.T)
X_dagger = np.linalg.pinv(X)
w_lin = X_dagger.dot(y)
# print('w_lin = ', w_lin)

# i = (X.dot(w_lin) - y)
E_in_sqr = 1/N * np.sum(np.square(X.dot(w_lin) - y))   
print('Q14, E_in_sqr = ', E_in_sqr)


# ----------------15--------------------
eta = 0.001
count_list = []
for i in range(1000):
    random.seed()
    w_t = np.zeros((w_lin.shape[0], 1))
    E_in_w_t = 1/N * np.sum(np.square(X.dot(w_t) - y))
    count = 0
    while not E_in_w_t <= 1.01 * E_in_sqr:
        rand = random.randrange(N)
        # print(rand)
        x_n = X[rand:rand+1, :].T
        y_n = y[rand]
        # print('x_n = ', x_n)
        # print(x_n.shape)
        # print('y_n = ', y_n)
        w_t = w_t + eta*2*(y_n - (w_t.T).dot(x_n)) * x_n
        E_in_w_t = 1/N * np.sum(np.square(X.dot(w_t) - y))
        count += 1
    print(i, end='\r')
    count_list.append(count)

print("Q15, average number of iteration = ", sum(count_list)/len(count_list))

# --------------16----------------

eta = 0.001
E_in_ce_list = []
for i in range(100):
    # print(f'number of iteration = {i}', end='\r')
    random.seed()
    w_t = np.zeros((w_lin.shape[0], 1))

    for i in range(500):
        rand = random.randrange(N)
        x_n = X[rand:rand+1, :].T
        y_n = y[rand]

        w_t = w_t + eta*(1/(1 + np.exp(y_n*(w_t.T).dot(x_n))))*y_n*x_n

    E_in_ce = 0

    E_in_ce = np.sum(np.log(np.ones((y.shape[0], 1)) + np.exp(-np.multiply(y, (X).dot(w_t)))))
    E_in_ce /= N
    E_in_ce_list.append(E_in_ce)

# print('', end='\r')
print("Q16, averaged cross-entropy error = ", sum(E_in_ce_list)/len(E_in_ce_list))

# -----------------17---------------

eta = 0.001
E_in_ce_list = []
for i in range(1000):
    # print(f'number of iteration = {i}', end='\r')
    random.seed()
    w_t = w_lin

    count = 0
    for i in range(500):
        rand = random.randrange(N)
        x_n = X[rand:rand+1, :].T
        y_n = y[rand]

        w_t = w_t + eta*(1/(1 + math.exp(y_n*(w_t.T).dot(x_n))))*y_n*x_n

        count += 1

    E_in_ce = 0

    for i in range(N):
        x_n = X[i:i+1, :].T
        y_n = y[i]
        E_in_ce += np.log(1 + math.exp(-y_n*(w_t.T).dot(x_n)))
    E_in_ce /= N
    E_in_ce_list.append(E_in_ce)

# # print('', end='\r')
print("Q17, average E_ce_in = ", sum(E_in_ce_list)/len(E_in_ce_list))

# ----------------------18----------------------
Num_in = 0
for i in range(N):
    if np.sign(X.dot(w_lin)[i]) != y[i]:
        Num_in += 1
E_in_01 = Num_in/N
print("Q18, E_in_01 = ", E_in_01)

Num_out = 0
for i in range(N_test):
    if np.sign(X_test.dot(w_lin)[i]) != y_test[i]:
        Num_out += 1
E_out_01 = Num_out/N_test
print("Q18, E_out_01 = ", E_out_01)

print("Q18, E_in_01 - E_out_01 = ", np.absolute(E_in_01 - E_out_01))

# --------------------19------------------
# print(X**3)
phi = np.c_[X, data[:, 1:-1]**2, data[:, 1:-1]**3]
# print(phi.shape)

phi_test = np.c_[X_test, data_test[:, 1:-1]**2, data_test[:, 1:-1]**3]

phi_dagger = np.linalg.pinv(phi)
w_lin = phi_dagger.dot(y)

Num_in = 0
for i in range(N):
    if np.sign(phi.dot(w_lin)[i]) != y[i]:
        Num_in += 1
E_in_01 = Num_in/N
print("Q19, E_in_01 = ", E_in_01)

Num_out = 0
for i in range(N_test):
    if np.sign(phi_test.dot(w_lin)[i]) != y_test[i]:
        Num_out += 1
E_out_01 = Num_out/N_test
print("Q19, E_out_01 = ", E_out_01)

print("Q19, E_in_01 - E_out_01 = ", np.absolute(E_in_01 - E_out_01))

# --------------20-------------------
phi = X
phi_test = X_test

for i in range(2, 11):
    phi = np.c_[phi, data[:, 1:-1]**i]
    phi_test = np.c_[phi_test, data_test[:, 1:-1]**i]

phi_dagger = np.linalg.pinv(phi)
w_lin = phi_dagger.dot(y)

Num_in = 0
for i in range(N):
    if np.sign(phi.dot(w_lin)[i]) != y[i]:
        Num_in += 1
E_in_01 = Num_in/N
print("Q20, E_in_01 = ", E_in_01)

Num_out = 0
for i in range(N_test):
    if np.sign(phi_test.dot(w_lin)[i]) != y_test[i]:
        Num_out += 1
E_out_01 = Num_out/N_test
print("Q20, E_out_01 = ", E_out_01)

print("Q20, E_in_01 - E_out_01 = ", np.absolute(E_in_01 - E_out_01))

