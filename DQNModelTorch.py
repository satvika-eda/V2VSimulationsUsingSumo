import torch
import torch.nn as nn
from simulation import *
import torch.optim as optim
import numpy as np
import random
import traci

class V2VState:
    def __init__(self, ego_vehicle, nearby_vehicles):
        self.ego_vehicle = ego_vehicle
        self.nearby_vehicles = nearby_vehicles

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


### reward 

class V2VRewards:
    def __init__(self, collision_penalty, speed_reward, end_reward, accel_change_penalty, contradicting_penalty):
        self.collision_penalty = collision_penalty
        self.speed_reward = speed_reward
        self.end_reward = end_reward
        self.accel_change_penalty = accel_change_penalty
        self.contradicting_penalty = contradicting_penalty
        # self.efficiency_reward = efficiency_reward

# Example Usage:
rewards = V2VRewards(-200, 10, 100, -5, -10)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.layer1 = nn.Linear(60, 5)
        # self.layer3 = nn.Linear(16, action_size)

    def forward(self, vehicle_states):
        vehicle_states[0] = np.delete(vehicle_states[0], 0)
        # print(vehicle_states[0])
        vehicles = np.array([vehicle_states[0]], dtype=np.float32)
        for i in vehicle_states[1]:
            i = np.delete(i, 0)
            np.append(vehicles, i)
        # print("mask b")
        # x = np.array([vehicle_states[0]])
        # for i in vehicle_states[1]:
        #     np.append(x, i)
        # print(vehicles)
        while vehicles.size != 60 :
            # print(len(vehicles))
            vehicles = np.append(vehicles, np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))
        # print("mask a")
        x = torch.tensor(vehicles)
        # print("I'm here a")
        x = torch.relu(self.layer1(x))
        # print("I'm here b")
        # x = torch.relu(self.layer2(x))
        x = torch.sigmoid(x)
        return x

        # network = Sequential()
        # network.add(Dense(24, input_dim = self.state_size, activation='relu'))
        # network.add(Dense(24, activation='relu'))
        # network.add(Dense(self.action_size, activation='sigmoid'))
        # network.compile(loss='mse', optimizer=adam_v2.Adam(lr=self.learning_rate))
        # return network

    def predict_action(self, state):
        values = self.forward(state)
        # print(values)
        action = []
        for i in values:
            if i > 0.5:
                action.append(True)
            else:
                action.append(False)
        # print("predict action")
        # print(action)
        return action

    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

num_episodes = 10
state_size = 10
action_size = 5
learning_rate = 0.001

model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def state_space_to_array(statespace):
    states = np.array([], dtype=np.float32)
    for i in statespace:
        array = np.array([], dtype=np.float32)
        array2 = np.array([], dtype=np.float32)
        x = np.delete(i[0], 0)
        for i in x:
            array2 = np.append(array2, float(i))
        array = np.append(array, array2)
        nearby_vehicles = np.array([], dtype=np.float32)
        for j in range(1, len(i[1])):
            if j%7!=0:
                nearby_vehicles = np.append(nearby_vehicles, float(i[1][j]))
        array = np.append(array, nearby_vehicles)
        states = np.append(states, array)
        # print(type(states[0]))
    return states

def train_dqn(model, batch_size):
    if len(model.memory) < batch_size:
        return
    
    batch = random.sample(model.memory, batch_size)
    states, actions, next_states, rewards, dones = zip(*batch)
    # print("gbhnjkl", states)
    # for i in states:
    #     print(i)
    states = state_space_to_array(states)
    print(type(states[8]))
    states = torch.tensor(states, dtype=torch.float64)
    actions = torch.tensor(actions, dtype=torch.long)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    # print(states)
    # print(actions)
    # print(next_states)
    # print(rewards)
    # print(dones)
    current_q = model(states).gather(1, actions.unsqueeze(1))
    next_q = model(next_states).max(dim=1)[0].unsqueeze(1)
    
    target_q = rewards + (1 - dones) * 0.99 * next_q
    loss = nn.MSELoss()(current_q, target_q.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 


for i in range(num_episodes):
    start_simulation()
    # print("simulation started")
    run_simulation(model) 
    # print("simulation ran as well") 
    train_dqn(model, 50)
    model.memory = []
traci.close()    

    






