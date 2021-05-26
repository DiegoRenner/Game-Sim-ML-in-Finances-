import numpy as np
from datetime import datetime
from tqdm import trange
import tensorflow as tf
import tensorflow_probability as tfp
from env_total_v2 import Game
from utils import create_input_batch, save_weights, load_weights


log_dir = "log/game/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# Hyperparameters
params = {
    "epochs_single": 5,  # number of epochs an agent is trained per training
    "epochs_total": 5,  # number of times each agent is trained
    "steps": 11,  # number of steps in each game
    "population_size": 20,  # number of different weights compared
    "max_iterations": 50,  # number of iterations optimization algorithm runs
    "agent_num": 2,
}
locals().update(params)

IRM_POOL = [0.01, 0.02, 0.05]
IRS_POOL = [0.0]
DIVM_POOL = [0.5, 1, 1.5]
DIVS_POOL = [0.2, 0.3, 0.4]
ENDOW_CASH_POOL = [20, 100, 80]
ENDOW_STOCK_POOL = [10, 15, 10]
save = False

# Batch of game scenarios
batch = create_input_batch(
    batch_size=epochs_single,
    agent_num=agent_num,
    steps=steps,
    irm_pool=IRM_POOL,
    divm_pool=DIVM_POOL,
    irs_pool=IRS_POOL,
    divs_pool=DIVS_POOL,
    endow_cash_pool=ENDOW_CASH_POOL,
    endow_stock_pool=ENDOW_STOCK_POOL,
)

game = Game(steps, agent_num)

# Dummy pass to ensure initialization
x = batch[0]
_ = game(x)


# Function to be optimized by differential evolution minimizer
def objective_fn(w1, b1, w2, b2):
    """
    Wrapper function around game for training a specific agent
    Each input is tensor; first dimension indexes the elements in the population
    Each population element contains weights / biases of layers of agent's NN
    """
    global iterations
    print(f"Training iteration {iterations}")
    iterations += 1

    for i in trange(len(w1), unit="population_element"):

        game.first_layers[training_agent].set_weights([w1[i], b1[i]])
        game.out_layers[training_agent].set_weights([w2[i], b2[i]])

        cumulative_reward = tf.Variable(tf.zeros(1))
        for epoch, game_inputs in enumerate(batch):
            _reward = game(game_inputs)[training_agent]
            cumulative_reward.assign_add(tf.expand_dims(_reward, 0))

        if i == 0:
            cumulative_rewards = cumulative_reward
        else:
            cumulative_rewards = tf.concat([cumulative_rewards, cumulative_reward], 0)

    return -cumulative_rewards  # minimize total negative rewards


# Train all agents
print("---------------------------------------")
print("Training output:")
for k in np.arange(epochs_total * game.agent_num):
    iterations = 0  # Reset iterations
    training_agent = k % game.agent_num  # Agent to be trained
    if training_agent == 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("EPOCHS_TOTAL: " + str(int(k / game.agent_num)))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(
        "####################### Training agent: "
        + str(training_agent)
        + " ######################"
    )

    # Initial weights of layers that are trained
    init_weights_first_layer = game.first_layers[training_agent].get_weights()
    init_weights_out_layer = game.out_layers[training_agent].get_weights()
    initial_position = [
        init_weights_first_layer[0],
        init_weights_first_layer[1],
        init_weights_out_layer[0],
        init_weights_out_layer[1],
    ]

    # Train one agent
    optim_results = tfp.optimizer.differential_evolution_minimize(
        objective_fn,
        initial_position=initial_position,
        population_size=population_size,
        max_iterations=max_iterations,
        seed=0,
    )

    # set weights of trained agent to best known
    game.first_layers[training_agent].set_weights(
        [optim_results.position[0], optim_results.position[1]]
    )
    game.out_layers[training_agent].set_weights(
        [optim_results.position[2], optim_results.position[3]]
    )

    # Evaluation
    for i, scenario in enumerate(batch):
        if i == 0:
            _rewards = game(scenario)
        else:
            _rewards += game(scenario)
    _rewards = _rewards / len(batch)

    for agent in np.arange(game.agent_num):
        with writer.as_default():
            tf.summary.scalar(f"Average reward {agent}", _rewards[agent], step=k)


if save:
    save_weights(
        game,
        f'saved_model_weights/weights_{datetime.now.strftime("%m%d%Y_%H%M%S")}.txt',
    )
