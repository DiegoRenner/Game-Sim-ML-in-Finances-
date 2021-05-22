import numpy as np
import datetime
from tqdm import trange
import tensorflow as tf
import tensorflow_probability as tfp
from env_total_v2 import Game
from utils import create_input_batch


# Hyperparameters
EPOCHS = 5
STEPS = 11
IRM_POOL = [0.01, 0.02, 0.05]
IRS_POOL = [0.0]
DIVM_POOL = [0.5, 1, 1.5]
DIVS_POOL = [0.2, 0.3, 0.4]
AGENT_NUM = 2
ENDOW_CASH_POOL = [20, 100, 80]
ENDOW_STOCK_POOL = [10, 15, 10]
POPULATION_SIZE = 50  # number of different weight specifications the opt. algorithm compares
MAX_ITERATIONS = 50  # number of iterations that the optimization algorithm runs
save_model = False

# Batch of game scenarios
batch = create_input_batch(batch_size=EPOCHS, agent_num=AGENT_NUM, steps=STEPS,
                           irm_pool=IRM_POOL, divm_pool=DIVM_POOL, irs_pool=IRS_POOL, divs_pool=DIVS_POOL,
                           endow_cash_pool=ENDOW_CASH_POOL, endow_stock_pool=ENDOW_STOCK_POOL)

game = Game(STEPS, AGENT_NUM)

# Summary and dummy pass to ensure initialization
x = batch[0]
_ = game(x)  
print(game.summary())

iterations = 0

# Training
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


training_agent = 0  # Agent to be trained


# Initial weights of layers that are trained
initial_position = [tf.convert_to_tensor(np.array(np.random.normal(0, 1, size=(1, 128)))),
                    tf.convert_to_tensor(np.zeros(128)),
                    tf.convert_to_tensor(np.array(np.random.normal(0, 1, size=(128, 1)))),
                    tf.convert_to_tensor(np.zeros(1))]

optim_results = tfp.optimizer.differential_evolution_minimize(objective_fn, initial_position=initial_position,
                                                              max_iterations=MAX_ITERATIONS, seed=0)
print(optim_results.converged)

# if save_model:
#   filename = 'weights_'
#   checkpoint_path = 'saved_model_weights/'
#   cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#           save_weights_only=True, verbose=1)
    
