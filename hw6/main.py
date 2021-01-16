import numpy as np
import random

# read data
data = np.genfromtxt('hw6_train.dat')
data_test = np.genfromtxt('hw6_test.dat')

# print(data.shape)
# print(data_test.shape)


def gini_calculator(data):
    N = len(data[:, 0])
    N_y_is_pos = np.count_nonzero(data[:, -1] == 1)
    N_y_is_neg = np.count_nonzero(data[:, -1] == -1)
    # print(N_y_is_pos, N_y_is_neg)
    # print(1 - (N_y_is_pos)^2 - (N_y_is_neg)^2)
    return 1 - (N_y_is_pos/N)**2 - (N_y_is_neg/N)**2


# generate a big number
branching_c_min = random.getrandbits(64)
i_min = None
theta_min = None

for i in range(len(data[0, :])-2, -1, -1):
    # print(i)
    data_sorted_i = data[data[:,i].argsort()]
    for theta in range(999, 0, -1):
        D1 = data_sorted_i[:theta, :]
        D2 = data_sorted_i[theta:, :]
        N1 = len(D1[:, 0])
        N2 = len(D2[:, 0])
        # print(N1, N2)
        gini_D1 = gini_calculator(D1)
        gini_D2 = gini_calculator(D2)
        branching_c = N1 * gini_D1 + N2 * gini_D2
        # print(branching_c)
        if branching_c <= branching_c_min:
            branching_c_min = branching_c
            i_min = i
            theta_min = theta

    data_sorted = data[data[:, i_min].argsort()]
    D1 = data_sorted[:theta_min, :]
    D2 = data_sorted[theta:, :]


print(i_min)
print(theta_min)
# print(data_sorted_i[999:,:].shape)

def DecisionTree(data):
    # Termination condition
    if np.all(data[:, -1] == 1) is True:

    
    # Learning creteria
    for i in range(len(data[0, :])-2, -1, -1):
        # print(i)
        data_sorted_i = data[data[:,i].argsort()]
        for theta in range(999, 0, -1):
            D1 = data_sorted_i[:theta, :]
            D2 = data_sorted_i[theta:, :]
            N1 = len(D1[:, 0])
            N2 = len(D2[:, 0])
            # print(N1, N2)
            gini_D1 = gini_calculator(D1)
            gini_D2 = gini_calculator(D2)
            branching_c = N1 * gini_D1 + N2 * gini_D2
            # print(branching_c)
            if branching_c <= branching_c_min:
                branching_c_min = branching_c
                i_min = i
                theta_min = theta

        # Split D into 2 parts
        data_sorted = data[data[:, i_min].argsort()]
        D1 = data_sorted[:theta_min, :]
        D2 = data_sorted[theta:, :]

        # Build sub-trees

class Node():
    def __init__(self, data):
        self.data = data
        self.i_min = None
        self.theta_min = None
        self.left = None
        self.right = None

def CART(node):
    if np.all(node.data[:, -1] == 1) is True:
        
    data = node.data
    for i in range(len(data[0, :])-2, -1, -1):
        data_sorted_i = data[data[:,i].argsort()]
        for theta in range(999, 0, -1):
            D1 = data_sorted_i[:theta, :]
            D2 = data_sorted_i[theta:, :]
            N1 = len(D1[:, 0])
            N2 = len(D2[:, 0])
            # print(N1, N2)
            gini_D1 = gini_calculator(D1)
            gini_D2 = gini_calculator(D2)
            branching_c = N1 * gini_D1 + N2 * gini_D2
            # print(branching_c)
            if branching_c <= branching_c_min:
                branching_c_min = branching_c
                i_min = i
                theta_min = theta

    node.left = Node(data[data[:,i_min].argsort()][:theta_min, :])
    node.right = Node(data[data[:,i_min].argsort()][theta_min:, :])
    node.i_min = i_min
    node.theta_min = (data[data[:,i].argsort()][theta-1, i] + data[data[:,i].argsort()][theta, i]) / 2
    return node


root = Node(data)