import numpy as np

# computes return for next step and uses it as input c for the exp_ut function
def returns_ut(a, r, d, D, w, s):
    c = r*w + d*s*D
    #return exp_ut(c,a)
    return exp_ut(c,a)

# takes risk aversion a and variable that the economic decision-maker prefers more of c
# return exponential utility U
def exp_ut(c, a):
    U = c
    if a != 0:
        U = 1 - np.exp(-c*a)/a
    return U