import torch
import torch.nn as nn
from simulation import *
import torch.optim as optim
import numpy as np
import random
import traci

# Representation of a V2V state (i.e. state of a vehicle and states of the nearby vehicles)
class V2VState:
    def __init__(self, ego_vehicle, nearby_vehicles):
        self.ego_vehicle = ego_vehicle
        self.nearby_vehicles = nearby_vehicles

# Representation of a vehicle state (i.e. State of a vehicle)
class VehicleState:
    def __init__(self, id, position, speed, acceleration, direction, lane):
        self.id = id
        self.position = position
        self.speed = speed
        self.acceleration = acceleration
        self.direction = direction
        self.lane = lane



### action representation
class V2VActions:
    def __init__(self, accelerate, decelerate, maintain_speed, change_lane, turn):
        self.accelerate = accelerate
        self.decelerate = decelerate
        self.maintain_speed = maintain_speed
        self.change_lane = change_lane
        self.turn = turn

# Example Usage:
# ego_vehicle_actions = V2VActions(accelerate=True, change_lane=True, turn=False)


### rewards and penalties
class V2VRewards:
    def __init__(self, collision_penalty, end_reward, contradicting_penalty, stop_penalty):
        self.collision_penalty = collision_penalty
        self.end_reward = end_reward
        self.contradicting_penalty = contradicting_penalty
        self.stop_penalty = stop_penalty

# Example Usage:
rewards = V2VRewards(-200, 500, -10, -50)

# Model network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []

        self.layer1 = nn.Linear(84, 32)
        self.layer2 = nn.Linear(32, 5)

    #Forward Propogation
    def forward(self, vehicle_states):
        x = torch.tensor(vehicle_states)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(x)
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

#Defining the state_size
state_size = 14
action_size = 5
learning_rate = 0.001

#Model
model = DQN(state_size, action_size)

#Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Retraining the model with the memory
def train_dqn(model, batch_size):
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

    loss = nn.MSELoss()(current_q, target_q.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    