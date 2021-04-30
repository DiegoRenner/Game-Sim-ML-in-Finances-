
# takes interest rate and current wealth
# returns wealth at next time step
def comp_int(r, wealth):
    wealth_updated = (1+r)*wealth
    return wealth_updated