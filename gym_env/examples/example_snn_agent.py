# Example of interacting with environment using Exponential Utility Agents, constant time series, equal endowments
from env import MarketEnv
from utils import create_constant_series, run
from snn_agent import ShallowNNAgent
import numpy as np
import pandas as pd

RUNS = 1
STEPS = 11
CONSTANT_INTEREST = 0.05
CONSTANT_DIVIDEND = 1
AGENT_NUM = 2
CASH_ENDOWMENT = 100
STOCK_ENDOWMENT = 10

interest_data, dividend_data = create_constant_series(CONSTANT_INTEREST, CONSTANT_DIVIDEND, STEPS)
agents = [ShallowNNAgent(i) for i in np.arange(AGENT_NUM)]
cash_endowments = [CASH_ENDOWMENT] * AGENT_NUM
stock_endowments = [STOCK_ENDOWMENT] * AGENT_NUM

params = [interest_data, dividend_data, agents, cash_endowments, stock_endowments]
trade_hist, final_states, final_rewards = run(RUNS, params)

print(f'Rewards: {final_rewards}')
