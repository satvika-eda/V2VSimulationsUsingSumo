import torch
import torch.nn as nn
from simulation import *
import torch.optim as optim
import numpy as np
import random
import traci
import matplotlib.pyplot as plt

# V2VState represents the state of the vehicle. It has its own attributes as well as the attributes of the nearby vehicles. 
class V2VState:
    def __init__(self, ego_vehicle, nearby_vehicles):
        self.ego_vehicle = ego_vehicle
        self.nearby_vehicles = nearby_vehicles

# VehicleState represents the paramets we are considering for the state of the vehicle. 
# id - id of the vehicle in the simulation
# position - To determining the distance between vehicles
# speed - To maintain relative speed among vehicles to avoid collisions
# acceleration - To either decelerate or accelerate to avoid collisions
# direction - To check if there are any other vehicles going in the same direction
# lane - To see if there are any vehicles stopped in the lane.
class VehicleState:
    def __init__(self, id, position, speed, acceleration, direction, lane):
        self.id = id
        self.position = position
        self.speed = speed
        self.acceleration = acceleration
        self.direction = direction
        self.lane = lane

### V2VActions represents the actions that can be performed on the vehicle for this project.
class V2VActions:
    def __init__(self, accelerate, decelerate, maintain_speed, change_lane, turn):
        self.accelerate = accelerate
        self.decelerate = decelerate
        self.maintain_speed = maintain_speed
        self.change_lane = change_lane
        self.turn = turn

### rewards and penalties
# V2VRewards represents the rewards and the penalities that are taken into consideration for this model.
# collision_penaltiy - penality when a collision happens in the simulation
# end_reward - reward is given, when vehicle reaches its end goal
# contradicting_penality - As we are performing more than 1 action at one step per vehicle, penality is given when the actions predicted are contradicting with each other like if accelerate and decelerate are given at the same time.
# stop_penality - when vehicle reaches stop state in simulation
class V2VRewards:
    def __init__(self, collision_penalty, end_reward, contradicting_penalty, stop_penalty):
        self.collision_penalty = collision_penalty
        self.end_reward = end_reward
        self.contradicting_penalty = contradicting_penalty
        self.stop_penalty = stop_penalty

# Below are the final rewards used for this model.
rewards = V2VRewards(-400, 1000, -80, -300)

# Model network
# we are using below DQN to train the model.
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        # Input layer the number of neurons depends on the number of vehicles and the parameters we are considering for each vehicle. 
        # In our project, there are 14 vehicles and each vehicle has 6 parameters so there are 84 neurons.
        self.layer1 = nn.Linear(state_size*6, 32)
        # The output layer consists of 5 neurons as we have 5 actions, we are using sigmoid activation function so the output values range from 0 to 1. 
        # As vehicles can perform more than 1 actions at the same simulation step, we have considered to perform actions on the vehicle that have more than the threshold value of 0.5.
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16,5)

    #Forward Propogation
    def forward(self, vehicle_states):
        x = torch.tensor(vehicle_states)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

    #Prediction
    def predict_action(self, state):
        values = self.forward(state)
        action = []
        for i in values:
            if i > 0.5:
                action.append(True)
            else:
                action.append(False)
        return action

    #List of states, actions, rewards, nest_states and dones.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# Defining the state_size, actioin_size and the learning rate to train the model
state_size = 14
action_size = 5
learning_rate = 0.00001

# Initializing the model
# model = DQN(state_size, action_size)

# Adam Optimizer


losses = []

# Retraining the model with the memory
def train_dqn(model, batch_size):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if len(model.memory) < batch_size:
        return
    
    batch = random.sample(model.memory, batch_size)
    states, actions, next_states, rewards, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    current_q = model(states).gather(1, actions)
    next_q = model(next_states)

    target_q = rewards.view(-1,1) + (1 - dones.view(-1,1)) * 0.99 * next_q

    loss = nn.CrossEntropyLoss()(current_q, target_q.detach())
    losses.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  

def graph():
    print(losses)
    plt.plot(losses)
    plt.savefig("lossCE.png")