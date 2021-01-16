import numpy as np

train = np.genfromtxt('hw6_train.dat')
x_train, y_train = train[:, :-1], train[:, -1]
test = np.genfromtxt('hw6_test.dat')
x_test, y_test = test[:, :-1], test[:, -1]

def gini(y):
    if (len(y) == 0):
        return 1
    t = np.mean(y == -1)
    
    return 1 - t**2 - (1 - t)**2

def lossfunc(theta, x, y):
    y1 = y[x < theta]
    y2 = y[x >= theta]
    Gini1 = gini(y1)
    Gini2 = gini(y2)
    
    return len(y1) * Gini1 + len(y2) * Gini2

def gen_theta(x):
    x = np.sort(x)
    theta = (x[1:] + x[:-1]) / 2

    return theta

def decision_stump(x, y):
    n, d = x.shape
    theta_min = 0
    d_min = 0
    branching_c_min = n
    for i in range(d):
        x_d = x[:, i]
        theta = gen_theta(x_d)
        for theta_ in theta:
            branching_c = lossfunc(theta_, x_d, y)
            if branching_c < branching_c_min:
                branching_c_min = branching_c
                theta_min = theta_
                d_min = i

    return d_min, theta_min, branching_c_min


def isstop(x, y):
    n = x.shape[0]
    n1 = np.sum(y != y[0])
    n2 = np.sum(x != x[0, :])
    return n1 == 0  or n2 == 0


class DTree:
    def __init__(self, theta, d, value=None):
        self.theta = theta
        self.d = d
        self.value = value
        self.left = None
        self.right = None

def learn_tree(x, y):
    if isstop(x, y):
        return DTree(None, None, y[0])
    else:
        d, theta, score = decision_stump(x, y)
        tree = DTree(theta, d)

        i1 = x[:, d] < theta
        x1 = x[i1]
        y1 = y[i1]
        
        i2 = x[:, d] >= theta
        x2 = x[i2]
        y2 = y[i2]

        tree.left = learn_tree(x1, y1)
        tree.right = learn_tree(x2, y2)
        return tree

def pred(tree, x):
    if tree.value != None:
        return tree.value
    if x[tree.d] < tree.theta:
        return pred(tree.left, x)
    else:
        return pred(tree.right, x)

def error(tree, x, y):
    ypred = [pred(tree, x_) for x_ in x]
    return np.mean(ypred != y)

def q14():
    dtree = learn_tree(x_train, y_train)
    print(error(dtree, x_train, y_train))
    print(error(dtree, x_test, y_test))

def q15():
    global train, test, x_train, y_train, x_test, y_test
    N_tree = 2000
    n = train.shape[0]
    E_out = 0
    for i in range(N_tree):
        index = np.random.randint(0, n, n//2)
        x_bag = x_train[index, :]
        y_bag = y_train[index]
        dtree = learn_tree(x_bag, y_bag)
        e_out = error(dtree, x_test, y_test)
        E_out += e_out
    E_out /= N_tree
    print(E_out)

def q16():
    global train, test, x_train, y_train, x_test, y_test
    N_tree = 2000
    n = train.shape[0]
    G = None
    for i in range(N_tree):
        index = np.random.randint(0, n, n//2)
        x_bag = x_train[index, :]
        y_bag = y_train[index]
        dtree = learn_tree(x_bag, y_bag)
        ypred = [pred(dtree, x_) for x_ in x_train]
        if i == 0:
            G = ypred
        else:
            G = [sum(x) for x in zip(G, ypred)]

    G = np.sign(G)
    E_in = np.mean(G != y_train)

    print(E_in)

def q17():
    global train, test, x_train, y_train, x_test, y_test
    N_tree = 2000
    n = train.shape[0]
    G = None
    for i in range(N_tree):
        index = np.random.randint(0, n, n//2)
        x_bag = x_train[index, :]
        y_bag = y_train[index]
        dtree = learn_tree(x_bag, y_bag)
        ypred = [pred(dtree, x_) for x_ in x_test]
        if i == 0:
            G = ypred
        else:
            G = [sum(x) for x in zip(G, ypred)]

    G = np.sign(G)
    E_out = np.mean(G != y_test)

    print(E_out)

def q18():
    global train, test, x_train, y_train, x_test, y_test
    N_tree = 2000
    n = train.shape[0]
    G = None
    oob_matrix = []
    for i in range(N_tree):
        oob_index = np.ones((n,), dtype = int)

        index = np.random.randint(0, n, n//2)

        for j in index:
            oob_index[j] = 0

        x_bag = x_train[index, :]
        y_bag = y_train[index]
        dtree = learn_tree(x_bag, y_bag)

        ypred = [pred(dtree, x_) for x_ in x_train]
        ypred = np.array(ypred)       
        ypred = ypred * oob_index

        if i == 0:
            G = ypred
        else:
            G += ypred
        
    # G = np.array(G)
    G[G == 0] = -1
    G = np.sign(G)
    print(G.shape)
    print(y_train.shape)
    E_oob = np.mean(G != y_train)
    # E_oob = np.sum(G) / n

    print(E_oob)
        

    




if __name__ == "__main__":
    q18()