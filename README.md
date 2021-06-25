# Game-Sim-ML-in-Finances


## Introduction
### The game in the field of RL
### Code structure

The code is mainly seperated into three parts: The game environment, the inupt data generation, and the training loop.

- Game Environment: Contains some parameters to define for instance how many timesteps to trade for, a relatively simple NN that takes as input the return on cash and stock investments and returns a price at which to buy/sell stocks, and a trading algorithm that based on prices computed by the NN and some randomness executes trades among all participating agents at each timestep. 

- Input Data Generation: Gets historical returns on cash and stocks from online sources and computes a horizon price using the Black-Scholes model. When a batch of multiple data sets is generated randomness is introduced by setting a random time window from which the historical data gets read.

- Training Loop: In each iteration of the training loop each agent gets trained separately. They are trained by optimizing the cumulative rewards over all datasets in the batch using the differential evolution minimizer from tensorflow. The reward for each agent is computed by taking their cash and adding the cash that comes from the number of owned stocks exchanged at the end of the game for the horizon price computed previously. The average price during trading, the average number of trades, and the average rewards for each agent over the entire batch are logged at every iteration and can later be viewed using tensorboard.

## Game description
### Trading algorithm

## Training
### Data generation
### Optimizer

## Results
