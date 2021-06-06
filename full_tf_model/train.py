import numpy as np
from datetime import datetime
from tqdm import trange
import tensorflow as tf
import tensorflow_probability as tfp
from utils import save_weights


def train(
    experiment,
    game,
    batch,
    epochs_total,
    population_size,
    max_iterations,
    save,
    evaluate_every=1,
):

    experiment.log_params_to_tensorboard()
    global iterations

    def objective_fn(w1, b1, w2, b2):
        """
        Wrapper function around game for training a specific agent
        Each input is tensor; first dimension indexes the elements in the population
        Each population element contains weights / biases of layers of agent's NN
        """
        global iterations
        iterations += 1
        t = trange(len(w1), unit="population_element")

        for i in t:
            t.set_description(f"iteration {iterations}")

            game.first_layers[training_agent].set_weights([w1[i], b1[i]])
            game.out_layers[training_agent].set_weights([w2[i], b2[i]])

            cumulative_reward = tf.Variable(tf.zeros(1))
            for epoch, game_inputs in enumerate(batch):
                _reward = game(game_inputs)[training_agent]
                cumulative_reward.assign_add(tf.expand_dims(_reward, 0))

            if i == 0:
                cumulative_rewards = cumulative_reward
            else:
                cumulative_rewards = tf.concat(
                    [cumulative_rewards, cumulative_reward], 0
                )

        return -cumulative_rewards  # minimize total negative rewards

    # Train all agents
    print("---------------------------------------")
    print("Training output:")
    for k in np.arange(epochs_total * game.agent_num):
        game.train = True
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
        if evaluate_every % k == 0:
            game.train = False
            logger_batch = []

            for i, scenario in enumerate(batch):
                _rewards = game(scenario)
                logger = game.logger
                logger["rewards"] = _rewards.numpy()
                logger["scenario"] = i
                logger_batch.append(logger)

            experiment.store_logger_batch(logger_batch, k)

    if save:
        filename = f'saved_model_weights/weights_{datetime.now.strftime("%m%d%Y_%H%M%S")}.txt'
        experiment.saved_model_weights = filename
        save_weights(game, filename)
