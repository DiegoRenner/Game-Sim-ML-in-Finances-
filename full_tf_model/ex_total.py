import tensorflow as tf
from env_total import Game
from utils import create_constant_series, create_input_tensor


ITERATIONS = 5
STEPS = 11
CONSTANT_INTERESTS = [0.01, 0.02, 0.05]
CONSTANT_DIVIDENDS = [0.5, 1, 1.5]
AGENT_NUM = 2
CASH_ENDOWMENTS = [20, 100, 80]
STOCK_ENDOWMENTS = [10, 15, 10]


game = Game(STEPS)

interest_data, dividend_data = create_constant_series(CONSTANT_INTERESTS[0], CONSTANT_DIVIDENDS[0], STEPS)
cash_endowment = [CASH_ENDOWMENTS[0]]*AGENT_NUM
stock_endowment = [STOCK_ENDOWMENTS[0]]*AGENT_NUM
x = create_input_tensor(interest_data, dividend_data, cash_endowment, stock_endowment)

game(x)
print(len(game.trainable_weights))


def training():  # TODO
    pass
