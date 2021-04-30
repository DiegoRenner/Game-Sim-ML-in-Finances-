import numpy as np

#initiate wealth uniformly distributed among agents
def init_w_uniform(N, n_t):
    w = np.zeros((n_t,N))
    w[0,:] = 10000
    return w
