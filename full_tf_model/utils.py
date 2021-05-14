import pandas as pd
import tensorflow as tf


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


def create_constant_series(interest_rate, dividend, steps):
    interest_data = [interest_rate] * steps
    dividend_data = [dividend] * steps
    return interest_data, dividend_data


def create_input_tensor(interest_data, dividend_data, endowments_cash, endowments_stock):
    return tf.convert_to_tensor(interest_data + dividend_data + endowments_cash + endowments_stock)
