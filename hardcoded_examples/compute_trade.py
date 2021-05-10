import numpy as np
import compute_utility as cu

# computes trades according to algorithm one of first write up
def compute_trade(N, r, d, w, s, D):
    w_new = np.copy(w)
    s_new = np.copy(s)

    # iterate over agnets in random order
    perm = np.random.permutation(N)
    for i in perm:
        U = cu.returns_ut(0,r,d,D,w_new,s_new)
        # select agent with lowest utility
        j = int(np.argmin(U))
        # determine if there will be trade
        if ((U[j] <= U[i]) & (i!=j)):
            # determine number of stocks agent i can buy
            stocks_to_buy = np.floor(w_new[i]/D)
            # trade depending on if there is more money than stock available or vice versa
            if  stocks_to_buy >= s_new[j]:
                s_new[i] += s_new[j]
                w_new[j] += D*s_new[j]
                w_new[i] -= D*s_new[j]
                s_new[j] = 0;
            else:
                s_new[i] += stocks_to_buy
                w_new[j] += D*stocks_to_buy
                w_new[i] -= D*stocks_to_buy
                s_new[j] -= stocks_to_buy
    return w_new, s_new
