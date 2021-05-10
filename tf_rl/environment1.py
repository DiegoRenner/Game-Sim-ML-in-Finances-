import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
import numpy as np
from tf_agents.trajectories import time_step as ts
import hardcoded_examples.initialize_simulation_size as iss
import hardcoded_examples.initialize_wealth as iw
import hardcoded_examples.initialize_stocks as ist
import hardcoded_examples.initialize_interest_and_dividends as iid

#initialize number of agents, stocks and timesteps
N, n, n_t = iss.init_sim_size_small()
# initialize risk free interest and dividend
r, d, D = iid.init_risk_div_const()
#initialize wealth/cash of each agent at each timestep
w = iw.init_w_uniform(N, n_t)
#stocks owned by each agent at each timestep
s = ist.init_s_uniform(N, n, n_t)

init_state = np.stack([w, s])
perm = np.random.permutation(N*N).reshape(N,N)
print(perm)
ratios = np.random.random(N*N).reshape(N, N)
action = np.stack([ratios.reshape(N, N), perm.reshape(N, N)]).reshape(2*N,N)
trade_seq = np.argsort(np.floor(100 * action[N:, :]), axis=None)


class GameSimEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
      shape=(2*N,N), dtype=np.float, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(2,N), dtype=np.float, minimum=0, name='observation')
    self._state = init_state
    self._episode_ended = False
    self._step_counter = 0

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = init_state
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    trade_seq = np.argsort(np.floor(100*action[N:,:])).reshape(N, N)
    trade_ratio = action[:N, :]

    # iterate over all possible buying/selling pairs
    for i in np.arange(0,N*N):

      # determine buyer seller and amount to trade
      buyer = int(trade_seq/N)
      seller = trade_seq%N
      trade_ratio_current = trade_ratio[buyer,seller]

      # stock to be taded
      s_tbt = min(np.floor(self._state[self._step_counter, buyer]/D),
                  self._state[N+self._step_counter, seller])
      w_tbt = s_tbt*D
      self._state[self._step_counter, buyer] -= w_tbt
      self._state[N+self._step_counter, buyer] += s_tbt
      self._state[self._step_counter, seller] += w_tbt
      self._state[N+self._step_counter, seller] -= s_tbt

    if self._step_counter == 10:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()
      self._episode_ended = True

    # Make sure episodes don't go on forever.kk
    if action == 1:
      self._episode_ended = True
    elif action == 0:
      new_card = np.random.randint(1, 11)
      self._state += new_card
    else:
      raise ValueError('`action` should be 0 or 1.')

    if self._episode_ended or self._state >= 21:
      reward =  np.sum(self._state[N-1,:]) + d*(1+r)/r*np.sum(self._state[2*N,:])
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
        np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)