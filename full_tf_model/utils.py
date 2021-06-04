import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import pandas_datareader
import quandl
from scipy.stats.mstats import gmean


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


def get_data():
    """get equal length dividend yield and interest rate time series for longest period possible
    dividend yield comes from S&P 500; interest rate from Kenneth French's database"""

    dy_monthly_data = quandl.get("MULTPL/SP500_DIV_YIELD_MONTH")["Value"]

    # Resampling yearly dividend yield and cutting of 2021
    dy_data = dy_monthly_data.resample("Y").apply(lambda x: gmean(x))[:-1]

    datareader = pandas_datareader.famafrench.FamaFrenchReader(
        "F-F_Research_Data_Factors", freq="Y", start=1926
    )
    int_data = datareader.read()[1]["RF"]  # 0 for monthly data; 1 for yearly data

    min_len = min(len(dy_data), len(int_data))

    return (int_data[-min_len:].values / 100), dy_data[-min_len:].values / 100


def get_sample(
    int_data, dy_data, steps=11, horizon=20, init_price=100, sigma=0.2, seed=38279
):
    """get a sample of interest rates and dividends from data containing
    interest rates and dividend yields
    stock price simulated using Black Scholes dynamics"""

    window_len = steps + horizon
    data_len = len(int_data)
    assert data_len >= window_len

    np.random.seed(seed)
    start = np.random.randint(low=0, high=data_len - window_len)

    interest_rates = int_data[start : start + window_len]
    dividend_yields = dy_data[start : start + window_len]

    random_factors = np.random.normal(size=window_len, scale=sigma)
    det_factors = interest_rates - dividend_yields

    prices = init_price * np.cumprod(1 + random_factors + det_factors)
    dividends = prices * dividend_yields

    discount_factors = 1 / np.cumprod(1 + interest_rates[-horizon:])
    horizon_price = np.dot(dividends[-horizon:], discount_factors)

    return interest_rates[:steps].tolist(), dividends[:steps].tolist(), horizon_price


def batch_from_data(
    int_data: list,
    dy_data: list,
    agent_num=2,
    steps=11,
    seeds=None,
    batch_size=5,
    endow_cash=500,
    endow_stock=5,
    horizon=20,
    init_price=100,
    sigma=0.2,
):

    assert (type(seeds) == None) or (type(seeds) == list)
    if seeds == None:
        seeds = [None] * batch_size

    endowments_cash = [endow_cash] * agent_num
    endowments_stock = [endow_stock] * agent_num

    batch = []
    for i in np.arange(batch_size):
        interest_rates, dividends, horizon_price = get_sample(
            int_data, dy_data, steps, horizon, init_price, sigma, seeds[i]
        )
        batch.append(
            create_input_tensor(
                interest_rates,
                dividends,
                endowments_cash,
                endowments_stock,
                horizon_price,
            )
        )
    return batch


def create_input_tensor(
    interest_data, dividend_data, endowments_cash, endowments_stock, horizon_price
):
    return tf.convert_to_tensor(
        interest_data
        + dividend_data
        + endowments_cash
        + endowments_stock
        + [horizon_price]
    )


def save_weights(game, filename):
    weights = []
    for i in np.arange(game.agent_num):
        first_layer_w, first_layer_b = game.first_layers[i].get_weights()
        out_layer_w, out_layer_b = game.out_layers[i].get_weights()
        weights.append([i, [first_layer_w, first_layer_b], [out_layer_w, out_layer_b]])
    with open(filename, "wb") as f:
        pickle.dump(weights, f)


def load_weights(game, filename):
    with open(filename, "rb") as f:
        weights = pickle.load(f)
    for i in np.arange(game.agent_num):
        game.first_layers[i].set_weights(weights[i][1])
        game.out_layers[i].set_weights(weights[i][2])
