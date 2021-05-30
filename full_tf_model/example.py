from env_total_v2 import Game
from utils import create_input_batch
from train import train


# Hyperparameters
game_params = {
    "end_time": 11,  # number of steps in each game
    "agent_num": 2,
}

training_params = {
    "epochs_total": 2,  # number of times each agent is trained
    "population_size": 5,  # number of different weights compared
    "max_iterations": 5,  # number of iterations optimization algorithm runs
    "log_path": "log/game/",
    "save": False,
}

batch_params = {
    "batch_size": 5,  # number of epochs an agent is trained per training
    "irm_pool": [0.01, 0.02, 0.05],
    "irs_pool": [0.0],
    "divm_pool": [0.5, 1, 1.5],
    "divs_pool": [0.2, 0.3, 0.4],
    "endow_cash_pool": [20, 100, 80],
    "endow_stock_pool": [10, 15, 10],
}

log_params = {**game_params, **training_params, "batch_size":
              batch_params["batch_size"]}

game = Game(**game_params)
batch = create_input_batch(game.agent_num, game.end_time, **batch_params)

# Dummy pass to ensure initialization
x = batch[0]
_ = game(x)

train(game, batch, **training_params, log_params=log_params)
