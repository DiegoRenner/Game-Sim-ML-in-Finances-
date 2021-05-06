# Example of interacting with environment using Exponential Utility Agents, constant time series, equal endowments

from env import Agent, MarketEnv
from utils import create_constant_series, ExpUtilityAgent
import numpy as np
import pandas as pd

EPISODES = 1
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

env = MarketEnv(interest_data=interest_data, dividend_data=dividend_data, agents=agents, endowments_cash=cash_endowments, endowments_stock=stock_endowments)

for i in range(STEPS):
    total_returns = env.get_returns()
    actions = [agent.act(c) for agent, c in zip(env.agents, total_returns)]
    state, done, rewards, info = env.step(actions)

for j in range(STEPS):
    print(f'Trades at time {j}:')
    for trade in env.trade_history[j]:
        print(f'Buyer: {trade.agent_buy}, Seller: {trade.agent_sell}, Quantity: {trade.quantity}, Price: {trade.price}')

print(f'Rewards: {rewards}')
