import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from simulationPPO import run_simulation, start_simulation


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
    def __init__(self, collision_penalty, speed_reward, end_reward, accel_change_penalty):
        self.collision_penalty = collision_penalty
        self.speed_reward = speed_reward
        self.end_reward = end_reward
        self.accel_change_penalty = accel_change_penalty
        # self.efficiency_reward = efficiency_reward

# Example Usage:
rewards = V2VRewards(-200, 10, 100, -5)

# Define your neural network architecture for the policy
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
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
    state_size =  10 # Define your state size
    action_size =  5 # Define your action size

    ppo = PPO(state_size, action_size)
    log_probs = []
    update_interval = 1

    for episode in range(10):
        start_simulation()
        run_simulation(ppo, rewards)
        # Get the current state from your V2V simulation
        # new_state =  #TODO new_state # Extract state from your simulation using V2VState

        # Get action and log probability from the PPO model
        # action, log_prob = ppo.get_action(state)

        # Execute the action in your simulation and get the reward
        # reward = #TODO calculate reward # Compute reward based on your V2VRewards

        # Store the state, action, log probability, and reward
        # states = new_state
        # actions.append(action)
        # log_probs.append(log_prob)
        # rewards.append(reward)

        # if episode % update_interval == 0:
        #     # Update the PPO model using stored experiences
        #     ppo.update_policy(states, actions, log_probs, rewards)
        #     states, actions, log_probs, rewards = [], [], [], []

main()