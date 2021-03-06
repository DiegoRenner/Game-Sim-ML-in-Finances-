import sys
import numpy as np
from sortedcontainers import SortedList
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from utils import Order, Trade


class Game(keras.Model):
    def __init__(self, end_time, agent_num, fc1_dims=128, init_bias=20.0):
        super(Game, self).__init__()

        self.time = 0
        self.end_time = end_time
        self.agent_num = agent_num
        self.fc1_dims = fc1_dims
        self.out_dims = end_time
        self.train = True
        self.logger = None

        # Tensors that unpack the input
        self.data = tf.Variable(tf.zeros(2 * self.end_time), trainable=False)
        self.book = tf.Variable(tf.zeros(2 * self.agent_num), trainable=False)
        self.horizon_price = tf.Variable(tf.zeros(1), trainable=False)

        # Layer that calculates the returns of each agent
        self.returns = tf.keras.layers.Lambda(self.get_returns)

        # Initializing the layers corresponding to different agents
        self.first_layers = [
            Dense(self.fc1_dims, activation="tanh") for i in np.arange(self.agent_num)
        ]
        initializer = tf.keras.initializers.Constant(init_bias)
        self.out_layers = [
            Dense(self.out_dims, bias_initializer=initializer)
            for i in np.arange(self.agent_num)
        ]

    def get_returns(self, input):
        """
        Calculates current (wrt. time) returns of the agents
        @param input: List containing the data and book tensors
        @return: tensor containing the returns
        """
        data, book = input[0], input[1]
        cash_return = tf.math.scalar_mul(data[self.time], book[: self.agent_num])
        div_return = tf.math.scalar_mul(
            data[self.end_time + self.time], book[self.agent_num :]
        )
        return tf.math.add(cash_return, div_return)

    def get_prices(self, returns):
        intermediate_outputs = [
            tf.Variable(tf.zeros(self.fc1_dims)) for i in np.arange(self.agent_num)
        ]
        prices = [tf.Variable(tf.zeros(1)) for i in np.arange(self.agent_num)]
        for i in np.arange(self.agent_num):
            intermediate_outputs[i].assign(
                tf.squeeze(self.first_layers[i](tf.reshape(returns[i], shape=(1, 1))))
            )
            prices[i].assign(
                tf.reshape(
                    self.out_layers[i](
                        tf.reshape(intermediate_outputs[i], shape=(1, self.fc1_dims))
                    )[0][self.time],
                    shape=(1,),
                )
            )
        return tf.nn.relu(prices)  # ensure non-negative prices

    def _payout_returns(self):
        returns = self.get_returns([self.data, self.book])
        stocks = self.book[self.agent_num :]
        self.book.assign(
            tf.concat(
                [tf.math.add(self.book[: self.agent_num], returns), stocks], axis=0
            )
        )

    def get_final_cash(self):
        return self.book[: self.agent_num] + tf.math.scalar_mul(
            tf.squeeze(self.horizon_price),
            self.book[self.agent_num : 2 * self.agent_num],
        )

    def _execute_trade(self, trade):
        _book = self.book.numpy()
        _book[trade.agent_buy] -= trade.price
        _book[trade.agent_sell] += trade.price
        _book[self.agent_num + trade.agent_buy] += trade.quantity
        _book[self.agent_num + trade.agent_sell] -= trade.quantity

        self.book.assign(tf.convert_to_tensor(_book))

    def trade(self):
        """Trading as described in Algorithm 1"""
        returns = self.returns([self.data, self.book])
        prices = self.get_prices(returns)
        orders = SortedList(key=lambda x: x.price)

        for i in np.arange(self.agent_num):
            orders.add(Order(agent_id=i, price=prices[i]))

        if not self.train:
            self.logger["initial orders"].append([order.numpy() for order in orders])

        trades = []
        while len(orders) > 1:
            # Draw random agent's action
            idx_bid = np.random.randint(len(orders))
            bid = orders.pop(idx_bid)

            cash_left = self.book[bid.agent_id].numpy()
            ask = orders[0]
            _seller_stocks = self.book[self.agent_num + ask.agent_id].numpy()

            if bid.price >= ask.price and cash_left >= ask.price and _seller_stocks > 1:
                _quantity = (
                    min(np.floor(cash_left / ask.price), _seller_stocks)
                    if ask.price > 0
                    else _seller_stocks
                )
                _trade = Trade(bid.agent_id, ask.agent_id, _quantity, ask.price)

                self._execute_trade(_trade)
                trades.append(_trade)

            # Comment the following lines to prevent recomputation of utilities during trading.
            returns = self.returns([self.data, self.book])
            prices = self.get_prices(returns)
            orders_temp = SortedList(key=lambda x: x.price)

            for i in np.arange(len(orders)):
                orders_temp.add(
                    Order(agent_id=orders[i].agent_id, price=prices[orders[i].agent_id])
                )
            orders = orders_temp

        if not self.train:
            self.logger["trades"].append(trades)

    def call(self, x):
        """
        Call function of the model
        @param x: tensor of interest_data, dividend_data, endowments_cash, endowments_stock
        @return: rewards of each agent
        """

        if not self.train:
            self.logger = {
                "scenario": None,
                "initial orders": [],
                "trades": [],
                "rewards": None,
            }

        # Unpacking the input tensor
        self.data.assign(x[: 2 * self.end_time])
        self.book.assign(x[2 * self.end_time : -1])
        self.horizon_price.assign(tf.expand_dims(x[-1], 0))

        for time in np.arange(self.end_time):
            self.time = time
            self.trade()
            self._payout_returns()

        return self.get_final_cash()  # Return final wealth as rewards
