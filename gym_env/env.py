import numpy as np
import gym
from sortedcontainers import SortedList


class MarketEnv(gym.Env):

    class Order:
        def __init__(self, agent_id, price):
            self.agent_id = agent_id
            self.price = price

    class Trade:
        def __init__(self, agent_buy, agent_sell, quantity, price):
            self.agent_buy = agent_buy
            self.agent_sell = agent_sell
            self.quantity = quantity
            self.price = price

    def __init__(self, interest_data, dividend_data, agents, endowments_cash, endowments_stock):
        super(MarketEnv, self).__init__()

        self.interest_data = interest_data  # Time series of the interest rate
        self.dividend_data = dividend_data  # Time series of dividends of stock
        self.time = 0  # Initialize time variable
        self.end_time = len(self.interest_data)  # Number of days in the data
        self.done = False
        self.agents = agents
        self.agent_num = len(self.agents)
        self.endowments_cash = endowments_cash  # List of cash endowments for each agent
        self.endowments_stock = endowments_stock  # List of stock endowments for each agent
        self.outstanding = sum(self.endowments_stock)  # Number of shares outstanding of the stock
        self.state_shape = 2 + 2 * self.agent_num
        self.rewards = [0] * self.agent_num
        self.trade_history = {}
        self.info = {}

        # Each agent: observes consumption and outputs utility
        self.observation_space = gym.spaces.Box(low=-1, high=np.inf, shape=(self.state_shape,))
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.agent_num,))

        # State tracks current interest rate, dividend and holdings of all agents
        self.state = [self.interest_data[self.time]] + [self.dividend_data[self.time]] \
                     + self.endowments_cash + self.endowments_stock

    def _execute_trade(self, trade):
        self.state[2 + trade.agent_buy] -= trade.price
        self.state[2 + trade.agent_sell] += trade.price
        self.state[2 + self.agent_num + trade.agent_buy] += trade.quantity
        self.state[2 + self.agent_num + trade.agent_sell] -= trade.quantity

    def get_final_wealth(self, agent_id):
        return self.state[2 + agent_id] + self.state[2 + self.agent_num + agent_id] * \
               self.state[1] * (self.state[0] + 1) / self.state[0]

    def get_returns(self):
        _wealth = np.array(self.state[2:2 + self.agent_num])
        _holdings = np.array(self.state[2 + self.agent_num:])
        total_returns = self.state[0] * _wealth + self.state[1] * _holdings
        return total_returns.tolist()

    def _payout_returns(self):
        self.state[2:2 + self.agent_num] = np.array(self.get_returns()) + \
                                           np.array(self.state[2:2 + self.agent_num]).tolist()

    def step(self, actions):
        """
        @param actions: utilites of agents sorted by agent ID (must be element in action space)
        @return: state, rewards, done, info
        """
        # Increase wealth of agents by bond return and dividend
        self._payout_returns()

        # Trading as described in Algorithm 1
        orders = SortedList(key=lambda x: x.price)

        for i in np.arange(self.agent_num):
            orders.add(self.Order(agent_id=i, price=actions[i]))

        trades = []

        while len(orders) > 1:
            # Draw random agent's action
            bid = np.random.choice(orders)
            orders.remove(bid)
            cash_left = self.state[2 + bid.agent_id]
            ask = orders[0]
            _seller_stocks = self.state[2 + self.agent_num + ask.agent_id]

            if bid.price >= ask.price and cash_left >= ask.price and _seller_stocks > 1:
                _quantity = min(np.floor(cash_left / ask.price), _seller_stocks)
                _trade = self.Trade(bid.agent_id, ask.agent_id, _quantity, ask.price)

                self._execute_trade(_trade)
                trades.append(_trade)

                if self.state[2 + self.agent_num + ask.agent_id] == 0:
                    orders.remove(ask)

        self.trade_history[self.time] = trades  # Store trades in environment

        # Step ends here. Now: Update parameters for next step
        self.time += 1
        self.done = self.time == self.end_time

        # Rewards zero apart from end of game
        if not self.done:
            self.state[0] = self.interest_data[self.time]
            self.state[1] = self.dividend_data[self.time]
            return self.state, self.done, self.rewards, self.info
        else:
            self.rewards = [self.get_final_wealth(agent_id) for agent_id in np.arange(self.agent_num)]
            return self.state, self.done, self.rewards, self.info
