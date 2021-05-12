from env import MarketEnv
import numpy as np
import pandas as pd


def create_constant_series(interest_rate, dividend, steps):
    interest_data = pd.Series([interest_rate] * steps)
    dividend_data = pd.Series([dividend] * steps)
    return interest_data, dividend_data


def run(runs, params):
    """Runs the game for specified number of runs without training."""
    trade_hist = []
    final_states = []
    final_rewards = []
    for i in np.arange(runs):
        env = MarketEnv(*params)
        for step in np.arange(env.end_time):
            total_returns = env.get_returns()
            actions = [agent.act([c]) for agent, c in zip(env.agents, total_returns)]
            print(f'ACTIONS: {actions}')
            state, done, rewards, info = env.step(actions)
        trade_hist.append(env.trade_history)
        final_states.append(state)
        final_rewards.append(rewards)
    return trade_hist, final_states, final_rewards
