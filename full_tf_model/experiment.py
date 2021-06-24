import numpy as np
import tensorflow as tf
import pickle
from tensorboard.plugins.hparams import api as hp


def get_trade_averages(logger_batch):
    trade_numbers = []
    trade_prices = []
    for logger in logger_batch:
        trades = [trade for l in logger["trades"] for trade in l]
        trade_numbers.append(len(trades))
        prices = [trade.price for trade in trades]
        trade_prices.append(np.mean(prices))
    avg_price = sum(trade_prices) / len(trade_prices)
    return sum(trade_numbers) / len(trade_numbers), avg_price


def get_average_rewards(logger_batch):
    for i, logger in enumerate(logger_batch):
        if i == 0:
            rewards = logger["rewards"]
        else:
            rewards += logger["rewards"]
    return rewards / len(logger_batch)


class Experiment:
    """Class for logging results of an experiment.
    Stores all logger dictionaries that are created when evaluating during
    the training loop.
    Can extend this class for all kind of summarizing metrics to be tracked.
    """

    def __init__(self, name, log_params, log_path):
        self.name = name
        self.log_params = log_params
        self.log_path = log_path
        self.logger_batches = {}
        self.epochs = 0
        self.saved_model_weights = None

        tb_dir = self.log_path + "tb/" + self.name
        self.writer = tf.summary.create_file_writer(tb_dir)

    def store_logger_batch(self, logger_batch: list, epoch: int):
        self.logger_batches[f"Epoch {epoch}"] = logger_batch
        self.epochs += 1
        self.write_to_tensorboard(logger_batch, epoch)

    def write_to_tensorboard(self, logger_batch, epoch):
        with self.writer.as_default():
            avg_trades, avg_price = get_trade_averages(logger_batch)
            tf.summary.scalar("Average number of trades", avg_trades, step=epoch)
            if avg_trades != 0:
                tf.summary.scalar("Average prices", avg_price, step=epoch)
            for agent in range(self.log_params["agent_num"]):
                avg_rewards = get_average_rewards(logger_batch)
                tf.summary.scalar(
                    f"Average reward agent {agent}", avg_rewards[agent], step=epoch
                )

    def log_params_to_tensorboard(self):
        with self.writer.as_default():
            hp.hparams(self.log_params)

    def save(self):
        self.writer = None
        #with open(self.log_path + "/" + self.name + ".pkl", "wb") as f:
        with open("log/" + self.name + ".pkl", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
