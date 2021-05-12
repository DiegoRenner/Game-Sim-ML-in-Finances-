import compute_utility as cu
import compute_trade as ct
import compute_interest as ci
import compute_dividends as cd

# run one simulation step according to the parameters set in run_simulation
# using hardcoded exponential utility function
def simulation_step(N, r, d, D, w, s):
    # compute wealtch after interest
    w_new = ci.comp_int(r,w)
    # compute dividends return
    w_new += cd.comp_div(d, D, s)
    # compute utility funcition
    U = cu.returns_ut(0,r,d,D,w,s)
    # execute trding
    w_new, s_new = ct.compute_trade_tf(N,r,d,w_new,s,D)
    return w_new, s_new

# run one simulation step according to the parameters set in run_simulation
# using tf NN as utility function
def simulation_step_tf(N, r, d, D, w, s):
    # compute wealtch after interest
    w_new = ci.comp_int(r,w)
    # compute dividends return
    w_new += cd.comp_div(d, D, s)
    # compute utility funcition
    U = cu.returns_ut(0,r,d,D,w,s)
    # execute trding
    w_new, s_new = ct.compute_trade_tf(N,r,d,w_new,s,D)
    return w_new, s_new
