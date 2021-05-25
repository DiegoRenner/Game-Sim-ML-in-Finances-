import numpy as np
from datetime import datetime
from tqdm import trange
import tensorflow as tf
import tensorflow_probability as tfp
from env_total_v2 import Game
from utils import create_input_batch, save_weights, load_weights


# Hyperparameters
EPOCHS_SINGLE = 5
EPOCHS_TOTAL = 5
STEPS = 11
IRM_POOL = [0.01, 0.02, 0.05]
IRS_POOL = [0.0]
DIVM_POOL = [0.5, 1, 1.5]
DIVS_POOL = [0.2, 0.3, 0.4]
AGENT_NUM = 2
ENDOW_CASH_POOL = [20, 100, 80]
ENDOW_STOCK_POOL = [10, 15, 10]
POPULATION_SIZE = 10  # number of different weight specifications compared
MAX_ITERATIONS = 0  # number of iterations that the optimization algorithm runs
save = False

# Batch of game scenarios
batch = create_input_batch(batch_size=EPOCHS_SINGLE, agent_num=AGENT_NUM, steps=STEPS,
                           irm_pool=IRM_POOL, divm_pool=DIVM_POOL, irs_pool=IRS_POOL, divs_pool=DIVS_POOL,
                           endow_cash_pool=ENDOW_CASH_POOL, endow_stock_pool=ENDOW_STOCK_POOL)

game = Game(STEPS, AGENT_NUM)

# Summary and dummy pass to ensure initialization
x = batch[0]
_ = game(x)
print(game.summary())

iterations = 0

# function to be optimized by differential evolution minimizer
def objective_fn(w1, b1, w2, b2):
    """
    Wrapper function around game for training a specific agent. Input to the optimization algorithm.
    each input is tensor where the first dimension indexes the elements in the population.
    each population element contains weights / biases of layers of agent's NN
    """
    global iterations
    print(f'Training iteration {iterations}')
    iterations += 1
    cumulative_rewards = tf.zeros(0, tf.float32)

    for i in trange(len(w1), unit='population_element'):

        game.first_layers[training_agent].set_weights([w1[i], b1[i]])
        game.out_layers[training_agent].set_weights([w2[i], b2[i]])

        cumulative_reward = tf.Variable(tf.zeros(1))
        for epoch, game_inputs in enumerate(batch):
            _reward = game(game_inputs)[training_agent]
            cumulative_reward.assign_add(tf.expand_dims(_reward, axis=0))

        cumulative_rewards = tf.concat([cumulative_rewards, cumulative_reward], axis=0)

    return -cumulative_rewards  # minimize total negative rewards


# Train all agents
print("---------------------------------------")
print("Training output:")
for k in np.arange(EPOCHS_TOTAL*AGENT_NUM):
    iterations = 0  # Reset iterations
    training_agent = k%AGENT_NUM  # Agent to be trained
    if training_agent == 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("EPOCHS_TOTAL: " + str(int(k/AGENT_NUM)))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("####################### Training agent: " + str(training_agent) + " ######################")
    # Initial weights of layers that are trained
    init_weights_first_layer = game.first_layers[training_agent].get_weights()
    init_weights_out_layer = game.out_layers[training_agent].get_weights()
    initial_position = [init_weights_first_layer[0],
                        init_weights_first_layer[1],
                        init_weights_out_layer[0],
                        init_weights_out_layer[1]]
    # Train one agent
    optim_results = tfp.optimizer.differential_evolution_minimize(objective_fn, 
        initial_position=initial_position, population_size=POPULATION_SIZE, 
        max_iterations=MAX_ITERATIONS, seed=0)

    # set weights of trained agent to best known
    game.first_layers[training_agent].set_weights([optim_results.position[0], optim_results.position[1]])
    game.out_layers[training_agent].set_weights([optim_results.position[2], optim_results.position[3]])

    # Output rewards of all agents given a sample input, for monitoring purposes
    print("Rewards for sample input:")
    output = game(x)
    for i in np.arange(AGENT_NUM):
        print("Agent " + str(i) +": " + str(output[i]))

if save:
    save_weights(game,
                 f'saved_model_weights/weights_{now.strftime("%m%d%Y_%H%M%S")}.txt')

