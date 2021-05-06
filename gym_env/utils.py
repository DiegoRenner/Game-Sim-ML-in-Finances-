from env import Agent, MarketEnv
import numpy as np
import pandas as pd


class ExpUtilityAgent(Agent):
    def __init__(self, agent_id, risk_aversion):
        super(ExpUtilityAgent, self).__init__(agent_id)
        self.risk_aversion = risk_aversion

    def act(self, consumption):
        if self.risk_aversion == 0:
            return consumption
        else:
            return (1 - np.exp(-self.risk_aversion * consumption)) / self.risk_aversion


def create_constant_series(interest_rate, dividend, steps):
    interest_data = pd.Series([interest_rate] * steps)
    dividend_data = pd.Series([dividend] * steps)
    return interest_data, dividend_data
