# Example of interacting with environment using Exponential Utility Agents, constant time series, equal endowments

from env import Agent, MarketEnv
from utils import create_constant_series, ExpUtilityAgent, run
import numpy as np
import pandas as pd

RUNS = 1
STEPS = 11
CONSTANT_INTEREST = 0.05
CONSTANT_DIVIDEND = 1
AGENT_NUM = 2
CASH_ENDOWMENT = 100
STOCK_ENDOWMENT = 10

RISK_AVERSIONS = np.random.random(AGENT_NUM) * 2

interest_data, dividend_data = create_constant_series(CONSTANT_INTEREST, CONSTANT_DIVIDEND, STEPS)
agents = [ExpUtilityAgent(i, RISK_AVERSIONS[i]) for i in np.arange(AGENT_NUM)]
cash_endowments = [CASH_ENDOWMENT] * AGENT_NUM
stock_endowments = [STOCK_ENDOWMENT] * AGENT_NUM

params = [interest_data, dividend_data, agents, cash_endowments, stock_endowments]
trade_hist, final_states, final_rewards = run(RUNS, params)

print(f'Rewards: {final_rewards}')
