import initialize_simulation_size as iss
import initialize_interest_and_dividends as iid
import initialize_wealth as iw
import initialize_stocks as ist
import simulation_step as ss

# initialize number of agents, stocks and timesteps
N, n, n_t = iss.init_sim_size_small()

# initialize risk free interest and dividend
r, d, D = iid.init_risk_div_const()

# initialize wealth/cash of each agent at each timestep
w = iw.init_w_uniform(N, n_t)
# stocks owned by each agent at each timestep
s = ist.init_s_uniform(N, n, n_t)

# run n_t simulation steps
for i in range(0, n_t - 1):
    print("--------------------------------------------------------------")
    print("Wealth at step " + str(i) + ": " + str(w[i, :]))
    print("Stocks at step " + str(i) + ": " + str(s[i, :]))
    w[i + 1, :], s[i + 1, :] = ss.simulation_step_tf(N, r, d, D, w[i, :], s[i, :])
