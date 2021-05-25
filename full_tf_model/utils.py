import numpy as np
import pandas as pd
import tensorflow as tf
import pickle


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


def create_random_series(ir_mean, div_mean, steps, ir_sigma=0, div_sigma=0):
    interest_data = np.random.normal(ir_mean, ir_sigma, steps).tolist()
    dividend_data = np.random.normal(div_mean, div_sigma, steps).tolist()
    return interest_data, dividend_data


def create_input_batch(batch_size: int, agent_num: int, steps: int, irm_pool: list, divm_pool: list,
                       irs_pool: list, divs_pool: list, endow_cash_pool: list, endow_stock_pool: list):
    """Creates @batch_size number of tensors that represent an input to a single game."""

    ir_means = np.random.choice(irm_pool, batch_size)
    div_means = np.random.choice(divm_pool, batch_size)
    ir_sigmas = np.random.choice(irs_pool, batch_size)
    div_sigmas = np.random.choice(divs_pool, batch_size)
    endowments_cash = np.random.choice(endow_cash_pool, batch_size)
    endowments_stock = np.random.choice(endow_stock_pool, batch_size)

    time_series_list = [create_random_series(ir_means[i], div_means[i], steps, ir_sigmas[i],
                                             div_sigmas[i]) for i in np.arange(batch_size)]

    # All agents have equal cash and stock endowments
    endowments_cash_list = [[endowments_cash[i]] * agent_num for i in np.arange(batch_size)]
    endowments_stock_list = [[endowments_stock[i]] * agent_num for i in np.arange(batch_size)]

    return [create_input_tensor(time_series_list[i][0], time_series_list[i][1],
                                endowments_cash_list[i], endowments_stock_list[i]) for i in np.arange(batch_size)]


def create_input_tensor(interest_data, dividend_data, endowments_cash, endowments_stock):
    return tf.convert_to_tensor(interest_data + dividend_data + endowments_cash + endowments_stock)


def save_weights(game, filename):
    weights = []
    for i in np.arange(game.agent_num):
        first_layer_w, first_layer_b = game.first_layers[i].get_weights()
        out_layer_w, out_layer_b = game.out_layers[i].get_weights()
        weights.append([i, [first_layer_w, first_layer_b],
                        [out_layer_w, out_layer_b]])
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)


def load_weights(game, filename):
    with open(filename, "rb") as f:
        weights = pickle.load(f)
    for i in np.arange(game.agent_num):
        game.first_layers[i].set_weights(weights[i][1])
        game.out_layers[i].set_weights(weights[i][2])

