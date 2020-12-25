import numpy as np
import random
from svmutil import *

with open('satimage.scale') as f:
    data = f.read()
with open('satimage.scale.t') as f1:
    data_test = f1.read()


 P15
y, x = svm_read_problem('satimage.scale')
print(type(y))
for i in range(len(y)):
    if y[i] != 3:
         y[i] = -1
    else:
        y[i] = 1

# m = svm_train(y, x, '-s 2 ')
prob = svm_problem(y, x)
param = svm_parameter('-s 0 -t 0 -c 10')
model = svm_train(prob, param)
support_vectors = model.get_SV()
support_vector_coefficients = model.get_sv_coef()
# print(support_vectors)
# print(support_vector_coefficients)

print(support_vector_coefficients[9][0])
w = np.zeros(36)
for i in range(len(support_vectors)):
    sv_vector = np.zeros(36)
    for j in range(36):
        try:
            sv_vector[j] = support_vectors[i][j+1]
        except:
            sv_vector[j] = 0
    w += np.multiply(sv_vector, support_vector_coefficients[i][0])
    
# print(support_vectors(35))
print(w)
print(np.sqrt(np.sum(np.square(w))))

# P16, P17
x, y = [0 for i in range(5)], [0 for i in range(5)]
for i in range(5):
    y, x = svm_read_problem('satimage.scale')
    for j in range(len(y)):
        if y[j] != i+1:
            y[j] = -1
        else:
            y[j] = 1

    # m = svm_train(y, x, '-s 2 ')
    prob = svm_problem(y, x)
    param = svm_parameter('-s 0 -t 1 -c 10 -d 2 -g 1 -r 1 -q')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y, x, model)
    support_vectors = model.get_SV()
    print("for N = {}".format(i+1))
    print(p_acc)
    print("Num of support vectors = ", len(support_vectors))


# P18
y, x = svm_read_problem('satimage.scale')
y_test, x_test = svm_read_problem('satimage.scale.t')
# print(type(y))

for i in range(len(y)):
    if y[i] != 6:
         y[i] = -1
    else:
        y[i] = 1

for i in range(len(y_test)):
    if y_test[i] != 6:
         y_test[i] = -1
    else:
        y_test[i] = 1

for i in [0.01, 0.1, 1, 10, 100]:
    prob = svm_problem(y, x)
    param = svm_parameter('-s 0 -t 2 -c {} -g 10 -q'.format(i))
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_test, x_test, model)

# P19
y, x = svm_read_problem('satimage.scale')
y_test, x_test = svm_read_problem('satimage.scale.t')
# print(type(y))

for i in range(len(y)):
    if y[i] != 6:
         y[i] = -1
    else:
        y[i] = 1

for i in range(len(y_test)):
    if y_test[i] != 6:
         y_test[i] = -1
    else:
        y_test[i] = 1

for i in [0.1, 1, 10, 100, 1000]:
    prob = svm_problem(y, x)
    param = svm_parameter('-s 0 -t 2 -c 0.1 -g {} -q'.format(i))
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_test, x_test, model)


# P20
y, x = svm_read_problem('satimage.scale')

for i in range(len(y)):
    if y[i] != 6:
         y[i] = -1
    else:
        y[i] = 1
# y_test, x_test = svm_read_problem('satimage.scale.t')
# print(type(y))
index_list = [i for i in range(len(y))]
gamma_candidate_dict = {}
for iter_ in range(1000):
    print("iter_{}".format(iter_))
    random.seed(iter_)
    index_val_list = random.sample(index_list, 200)
    index_val_list.sort()
    index_train_list = list(set(index_list) - set(index_val_list))

    y_val = [y[i] for i in index_val_list]
    x_val = [x[i] for i in index_val_list]

    y_train = [y[i] for i in index_train_list]
    x_train = [x[i] for i in index_train_list]

    p_acc_max = 0
    gamma_candidate = 0
    gamma_list = [0.1, 1, 10, 100, 1000] 
    gamma_list.reverse() # for the sake of choosing the smaller gamma when the p_acc's are tied
    for i in gamma_list:
        prob = svm_problem(y_train, x_train)
        param = svm_parameter('-s 0 -t 2 -c 0.1 -g {} -q'.format(i))
        model = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(y_val, x_val, model)
        # print('p_acc = ', p_acc[0])
        if p_acc[0] >= p_acc_max:
            p_acc_max = p_acc[0]
            gamma_candidate = i
        # print(gamma_candidate)
    
    if gamma_candidate not in gamma_candidate_dict:
        gamma_candidate_dict[gamma_candidate] = 1
    else:
        gamma_candidate_dict[gamma_candidate] += 1

    print("gamma_candidate_dict = ", gamma_candidate_dict)
    



# for i in range(len(y)):
#     if y[i] != 6:
#          y[i] = -1
#     else:
#         y[i] = 1

# for i in range(len(y_test)):
#     if y_test[i] != 6:
#          y_test[i] = -1
#     else:
#         y_test[i] = 1

# for i in [0.1, 1, 10, 100, 1000]:
#     prob = svm_problem(y, x)
#     param = svm_parameter('-s 0 -t 2 -c 0.1 -g {} -q'.format(i))
#     model = svm_train(prob, param)
#     p_label, p_acc, p_val = svm_predict(y_test, x_test, model)