import numpy as np

#initiate stocks uniformly distributed among agents
def init_s_uniform(N, n, n_t):
    s = np.zeros((n_t,N))
    s[0,:] = np.floor(n/N)
    return s