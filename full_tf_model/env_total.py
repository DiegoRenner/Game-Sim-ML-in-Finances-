import os
import numpy as np
from sortedcontainers import SortedList
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from utils import Order, Trade


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Game(keras.Model):

    def __init__(self, end_time, num_agents, fc1_dims=128):
        super(Game, self).__init__()

        self.time = 0
        self.end_time = end_time
        self.agent_num = num_agents
        self.fc1_dims = fc1_dims
        self.out_dims = 1

        # Tensors that unpack the input
        self.data = tf.Variable(tf.zeros(2 * self.end_time), trainable=False)
        self.book = tf.Variable(tf.zeros(2 * self.agent_num), trainable=False)

        # Layer that calculates the returns of each agent
        self.returns = tf.keras.layers.Lambda(self.get_returns)

        # NN corresponding to first agent
        self.first_layers = [Dense(self.fc1_dims, activation='relu')]*self.agent_num
        for i in np.arange(self.agent_num):
            self.first_layers[i] = Dense(self.fc1_dims, activation='relu')
        self.out_layers = [Dense(self.out_dims)]*self.agent_num
        for i in np.arange(self.agent_num):
            self.out_layers[i] = Dense(self.out_dims)
        #self.nn1_fc1 = Dense(self.fc1_dims, activation='relu')
        #self.nn1_out = Dense(self.out_dims)

        ## NN corresponding to second agent
        #self.nn2_fc1 = Dense(self.fc1_dims, activation='relu')
        #self.nn2_out = Dense(self.out_dims)

    def get_returns(self, input):
        """
        Calculates current (wrt. time) returns of the agents
        @param input: List containing the data and book tensors
        @return: tensor containing the returns
        """
        data, book = input[0], input[1]
        cash_return = tf.math.scalar_mul(data[self.time], book[:self.agent_num])
        div_return = tf.math.scalar_mul(data[self.end_time + self.time], book[self.agent_num:])
        return tf.math.add(cash_return, div_return)

    def _payout_returns(self, input):
        data, book = input[0], input[1]
        returns = self.get_returns(input)
        book[:self.agent_num] = tf.math.add(book[:self.agent_num], returns)
        return book

    def get_final_cash(self):
        r = self.data[self.end_time - 1].numpy()
        d = self.data[-1].numpy()
        rate = np.multiply(d, np.divide(1 + r, r))
        return self.book[:self.agent_num] + tf.math.scalar_mul(rate, self.book[self.agent_num:2 * self.agent_num])

    def _execute_trade(self, trade):
        _book = self.book.numpy()
        _book[trade.agent_buy] -= trade.price
        _book[trade.agent_sell] += trade.price
        _book[self.agent_num + trade.agent_buy] += trade.quantity
        _book[self.agent_num + trade.agent_sell] += trade.quantity

        self.book.assign(tf.convert_to_tensor(_book))

    def trade(self, prices):
        """Trading as described in Algorithm 1"""
        orders = SortedList(key=lambda x: x.price)

        for i in np.arange(self.agent_num):
            orders.add(Order(agent_id=i, price=prices[i]))

        trades = []

        while len(orders) > 1:
            # Draw random agent's action
            bid = np.random.choice(orders)
            orders.remove(bid)
            cash_left = self.book[bid.agent_id].numpy()
            ask = orders[0]
            _seller_stocks = self.book[self.agent_num + ask.agent_id].numpy()

            if bid.price >= ask.price and cash_left >= ask.price and _seller_stocks > 1:
                _quantity = min(np.floor(cash_left / ask.price), _seller_stocks)
                _trade = Trade(bid.agent_id, ask.agent_id, _quantity, ask.price)

                self._execute_trade(_trade)
                trades.append(_trade)

                if self.book[self.agent_num + ask.agent_id] == 0:
                    orders.remove(ask)

    def call(self, x):
        """
        Call function of the model
        @param x: tensor of interest_data, dividend_data, endowments_cash, endowments_stock
        @return: rewards of each agent
        """

        # Unpacking the input tensor
        self.data.assign(x[:2*self.end_time])
        self.book.assign(x[2*self.end_time:])

        # Initializing the tensors with indifference prices with dummy values
        p1 = tf.Variable(tf.convert_to_tensor(1.0))
        p2 = tf.Variable(tf.convert_to_tensor(1.0))

        for time in np.arange(self.end_time):
            self.time = time

            # Calculate returns
            returns = self.returns([self.data, self.book])

            # Forward pass of returns through the NN of each agent
            _nn = [tf.zeros((self.fc1_dims,))]*self.agent_num
            for i in np.arange(self.agent_num):
                _nn[i] = self.first_layers[i](tf.reshape(returns[i], shape=(1, 1)))

            p = [tf.Variable(tf.convert_to_tensor(1.0))]*self.agent_num
            for i in np.arange(self.agent_num):
                p[i].assign(tf.squeeze(self.out_layers[i](_nn[i])))

            # Trading
            self.trade(p)

            self.book.assign(self._payout_returns([self.data, self.book]))  # Payout returns

        return self.get_final_cash()  # Return final wealth as rewards
