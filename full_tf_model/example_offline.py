import pickle
from env_total_v2 import Game
from utils import get_data_offline, batch_from_data
from train import train
from experiment import Experiment


# Hyperparameters
game_params = {
    "end_time": 11,  # number of steps in each game
    "agent_num": 2,
}

training_params = {
    "epochs_total": 10,  # number of times each agent is trained
    "population_size": 10,  # number of different weights compared
    "max_iterations": 10,  # number of iterations optimization algorithm runs
    "save": False,
    "evaluate_every": 1,
}

batch_params = {
    "batch_size": 10,  # number of epochs an agent is trained per training
    "endow_cash": 500,
    "endow_stock": 5,
    "horizon": 20,
    "init_price": 100,
    "sigma": 0.2,
}

seeds = [23462, 192, 3817, 9732, 4, 55, 239, 11203, 2, 900,
         9209 ,12399, 29, 192, 877, 20, 5001, 230, 100, 1492]

log_params = {**game_params, **training_params, **batch_params}
experiment = Experiment("Example", log_params, "log/example/")

int_data, dy_data = get_data_offline()
game = Game(**game_params)
batch = batch_from_data(
    int_data, dy_data, game.agent_num, game.end_time, seeds, **batch_params
)

# Dummy pass to ensure initialization
x = batch[0]
_ = game(x)

train(experiment, game, batch, **training_params)
experiment.save()

# load experiment
# with open("log/example/Example.pkl", 'rb') as f:
#     experiment = pickle.load(f)
