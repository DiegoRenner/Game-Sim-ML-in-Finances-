import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
import numpy as np
from tf_agents.trajectories import time_step as ts
import hardcoded_examples.initialize_simulation_size as iss
import hardcoded_examples.initialize_wealth as iw
import hardcoded_examples.initialize_stocks as ist
import hardcoded_examples.simulation_step as ss
import hardcoded_examples.initialize_interest_and_dividends as iid

#initialize number of agents, stocks and timesteps
N, n, n_t = iss.init_sim_size_small()
#initialize wealth/cash of each agent at each timestep
w = iw.init_w_uniform(N, n_t)
# initialize risk free interest and dividend
r, d, D = iid.init_risk_div_const()
#stocks owned by each agent at each timestep
s = ist.init_s_uniform(N, n, n_t)

init_state = np.stack([w[0,:], s[0,:]])
perm = np.random.permutation(N*N).reshape(N,N)
print(perm)
ratios = np.random.random(N*N).reshape(N, N)
action = np.stack([ratios.reshape(N, N), perm.reshape(N, N)]).reshape(2*N,N)
trade_seq = np.argsort(np.floor(100 * action[N:, :]), axis=None)


class GameSimEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(N,), dtype=np.float, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(2,N), dtype=np.float, minimum=0, name='observation')
    self._state = init_state
    self._step_counter = 0

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = init_state
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.float))

  def _step(self, action):

    #trade_seq = np.argsort(np.floor(100*action[N:,:])).reshape(N, N)
    #trade_ratio = action[:N, :]


    # The last action ended the episode. Ignore the current action and start
    # a new episode.
    if self._step_counter == 10:
      return self.reset()

    w[self._step_counter + 1, :], s[self._step_counter + 1, :] = \
      ss.simulation_step(N, r, d, D, w[self._step_counter, :], s[self._step_counter, :])


