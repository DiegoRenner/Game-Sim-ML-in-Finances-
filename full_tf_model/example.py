import pickle
import numpy as np
from env_total_v2 import Game
from utils import get_data_online, get_data_offline, batch_from_data
from train import train
from experiment import Experiment
from datetime import datetime


run_offline = True

# Hyperparameters
game_params = {
    "end_time": 11,  # number of steps in each game
    "agent_num": 2,
    "fc1_dims": 128,
    "init_bias": 20,
}

training_params = {
    "epochs_total": 2,  # number of times each agent is trained
    "population_size": 5,  # number of different weights compared
    "population_stddev": 1.0,
    "max_iterations": 2,  # number of iterations optimization algorithm runs
    "differential_weight": 0.5,
    "crossover_prob": 0.9,
    "save": False,
    "evaluate_every": 1,
}

batch_size = 10
seeds = np.random.randint(0, 10000, batch_size)
seeds_log = {"seeds": np.array2string(seeds)}
seeds = np.random.randint(0, 10000, batch_size).tolist()

batch_params = {
    "batch_size": batch_size,  # number of epochs an agent is trained per training
    "endow_cash": 500,
    "endow_stock": 5,
    "horizon": 20,
    "init_price": 100,
    "sigma": 0.2,
}


log_params = {**game_params, **training_params, **batch_params, **seeds_log}
experiment_handle = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
experiment = Experiment(
    "Example_" + experiment_handle, log_params, "log/example_" + experiment_handle
)

if run_offline:
    int_data, dy_data = get_data_offline()
else:
    int_data, dy_data = get_data_online()

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
