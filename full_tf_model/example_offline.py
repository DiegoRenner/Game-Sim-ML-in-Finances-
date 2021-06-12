from env_total_v2 import Game
from utils import get_data_offline, get_data_online, batch_from_data, save_data
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
    "endow_cash": 500,
    "endow_stock": 5,
    "horizon": 20,
    "init_price": 100,
    "sigma": 0.2,
}

seeds = [23462, 192, 3817, 9732, 4]

log_params = {**game_params, **training_params, **batch_params}
int_data, dy_data = get_data_offline()

game = Game(**game_params)
batch = batch_from_data(
    int_data, dy_data, game.agent_num, game.end_time, seeds, **batch_params
)

# Dummy pass to ensure initialization
x = batch[0]
_ = game(x)

train(game, batch, **training_params, log_params=log_params)
