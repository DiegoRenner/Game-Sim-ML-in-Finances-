# Game-Sim-ML-in-Finances


## Introduction
### The game in the field of RL
### Code structure

The code is mainly seperated into three parts: The game environment, the inupt data generation, and the training loop.

- Game Environment: Contains some parameters to define for instance how many timesteps to trade for, a relatively simple NN that takes as input the return on cash and stock investments and returns a price at which to buy/sell stocks, and a trading algorithm that based on prices computed by the NN and some randomness executes trades among all participating agents at each timestep. 

- Input Data Generation: Gets historical returns on cash and stocks from online sources and computes a horizon price using the Black-Scholes model. When a batch of multiple data sets is generated randomness is introduced by setting a random time window from which the historical data gets read.

- Training Loop: In each iteration of the training loop each agent gets trained separately. They are trained by optimizing the cumulative rewards over all datasets in the batch using the differential evolution minimizer from tensorflow. The reward for each agent is computed by taking their cash and adding the cash that comes from the number of owned stocks exchanged at the end of the game for the horizon price computed previously. The average price during trading, the average number of trades, and the average rewards for each agent over the entire batch are logged at every iteration and can later be viewed using tensorboard.

## Game description
The game contains parameters on how many timesteps to trade for, the number of agents trading, a NN that determines the prices these agents trade at and a trading algorithm that executes the trading at each timestep which are looped over within the game as well.

### Trading algorithm
The trading algorithm iterates over each agent in a random order. In each iteration it then takes the agent (agent 1) besides the randomly chosen one (agent 0) that offers the lowest price for their stock. If the ask price from agent 1 is lower than the bid price from agent 0, agent 0 then buys as much of the stock from agent 1 as his cash allows. Agent 0 is now removed from the pool of agents and the next iteration begins by choosing the next agent at random.

### Paying out the final reward
The final reward is payed out as all the cash an agent holds plus all of their stock exchanged for cash at the horizon price computed in the input data generation.

## Training
### Data generation
### Optimizer

## Results
To keep things simple we started out only training a scenario with two agents in order to see in convergence was even possible for the simplest case.
We could see that starting our training with the parameters defined in example.py the average prices for which the stocks were trading would converge somewhere around 175.

In this example we can see that starting to train with prices above 200 would have them fall steadily:
https://tensorboard.dev/experiment/y50sBqELRmGPzvvI7MyEFw/#scalars

In another example we saw that starting with prices around 100 would have them rise up steadily during the training:
https://tensorboard.dev/experiment/ABC9UZErTGGaZmKwNj0f1w/#scalars
https://tensorboard.dev/experiment/MwzByVCDQOO2ofgek2yqsg/#scalars (longer)

Setting the prices at 175 in the beginning would let us see that they would stay around this range:
https://tensorboard.dev/experiment/Ln4K7bKCRkKrbxzo9bzOjA/#scalars

We could however also see in this previous example that this seems to be a local minimum where one of the agents gets the upper hand, always increasing it's reward when it's turn to optimize comes around. The other agent however doesn't manage to gain back all of the advantage in it's own turn.