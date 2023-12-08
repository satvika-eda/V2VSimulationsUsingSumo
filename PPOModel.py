import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from simulationPPO import run_simulation, start_simulation
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

### reward 

class V2VRewards:
    def __init__(self, collision_penalty, end_reward, contradicting_penalty, stop_penalty):
        self.collision_penalty = collision_penalty
        # self.speed_reward = speed_reward
        self.end_reward = end_reward
        # self.accel_change_penalty = accel_change_penalty
        self.contradicting_penalty = contradicting_penalty
        self.stop_penalty = stop_penalty
        # self.efficiency_reward = efficiency_reward

# Example Usage:
rewards = V2VRewards(-400, 300, -80, -500)

# Define your neural network architecture for the policy
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(x)
        return x

# Proximal Policy Optimization algorithm
class PPO:
    def __init__(self, state_size, action_size):
        self.policy = Policy(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.eps_clip = 0.2

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def update_policy(self, states, actions, log_probs, rewards):
        returns = self.compute_returns(rewards)
        returns = torch.tensor(returns, dtype=torch.float32)

        old_probs = torch.exp(log_probs)

        for _ in range(3):  # Update the policy for a few epochs
            new_probs = self.policy(states)
            new_probs = new_probs.gather(1, actions.unsqueeze(1))

            ratio = new_probs / (old_probs + 1e-5)
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * returns
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns
    

# Main simulation loop
def main():
    # Define state and action sizes based on your representations
    state_size =  14 # Define your state size
    action_size =  5 # Define your action size

    ppo = PPO(state_size, action_size)
    start_simulation()
    for _ in range(10):
        run_simulation(ppo, rewards)
    traci.close()

main()