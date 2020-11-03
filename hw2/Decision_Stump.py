import numpy as np
import random



def decision_stump(x, size, tau):
    # print("x = ", x)

    theta = [-1]
    for i in range (len(x) - 1):
        theta.append((x[i] + x[i+1])/2)

    # print("theta = ", theta)

    s = [-1, 1]

    error_min = 1
    s_g = 0
    t_g = 0

    f = [i if i != 0 else -1 for i in np.sign(x)]
    f_tau = []
    for ff in f:
        random.seed()
        prob = random.uniform(0, 1)
        # print("prob = ", prob)
        if prob <= tau:
            ff *= -1
        f_tau.append(ff)

    for ss in s:
        for tt in theta:
            error = 0
            h = [i if i/ss != 0 else -ss for i in ss*np.sign(x-tt)] # need to address to sign(0) to -1
            # f = [i if i != 0 else -1 for i in np.sign(x)]
            # f = [x if random.random() >= tau else -x for x in f]


            # print("h, f, f_tau = ", h, f, f_tau)

            for hh, ff in zip(h, f_tau):
                if hh != ff:
                    error += 1
            
            error /= size
            # print("error = ", error)

            if error < error_min:
                error_min = error
                s_g = ss
                t_g = tt
            # print("s_g, t_g = ", s_g, t_g)

    E_out = 0.5*abs(t_g) if s_g == 1 else 1 - 0.5*abs(t_g)
    E_out = ((1 - 2*tau)*E_out + tau)
    E_in = error_min
    # print("E_out, E_in = ", E_out, E_in)
    return E_out - E_in

