import compute_utility as cu
import compute_trade as ct
import compute_interest as ci
import compute_dividends as cd

# run one simulation step according to the parameters set in run_simulation
def simulation_step(N, r, d, D, w, s):
    # compute wealtch after interest
    w_new = ci.comp_int(r,w)
    # compute dividends return
    w_new += cd.comp_div(d, D, s)
    # compute utility funcition
    U = cu.returns_ut(0,r,d,w,s)
    # execute trding
    w_new, s_new = ct.compute_trade(N,r,d,w_new,s,D)
    return w_new, s_new