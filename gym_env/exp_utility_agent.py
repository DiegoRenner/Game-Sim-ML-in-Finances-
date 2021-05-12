from agent import Agent
import numpy as np


class ExpUtilityAgent(Agent):
    def __init__(self, agent_id, risk_aversion):
        super(ExpUtilityAgent, self).__init__(agent_id)
        self.risk_aversion = risk_aversion

    def act(self, consumption):
        consumption = consumption[0]
        if self.risk_aversion == 0:
            return consumption
        else:
            return (1 - np.exp(-self.risk_aversion * consumption)) / self.risk_aversion
