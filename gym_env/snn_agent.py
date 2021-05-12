from agent import Agent
from shallow_nn import ShallowNN
import tensorflow as tf
import numpy as np


class ShallowNNAgent(Agent):
    def __init__(self, agent_id):
        super(ShallowNNAgent, self).__init__(agent_id)
        self.network = ShallowNN()

    def act(self, observation):
        state = tf.convert_to_tensor([observation])
        print(state)
        return self.network(state).numpy()[0][0]

    def train(self, state):
        pass
