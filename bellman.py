import traci
import sumolib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random

# Define the SUMO network and route files
net_file = 'demo2.net.xml'
rou_file = 'demo2.rou.xml'
sumoBinary = sumolib.checkBinary("sumo-gui")
# Connect to SUMO using TraCI
traci.start([sumoBinary, "-c", "demo2.sumocfg","--start", "--collision.stoptime", "5", "--time-to-teleport", "-2",])

def add_aggressive_behavior(veh_id):
    traci.vehicle.setAccel(veh_id, 10)
    traci.vehicle.setDecel(veh_id, 4)
    traci.vehicle.setEmergencyDecel(veh_id, 0.1)
    traci.vehicle.setApparentDecel(veh_id, 0.1)
    traci.vehicle.setTau(veh_id, 0.01)
    traci.vehicle.setImperfection(veh_id, 0.1)
    traci.vehicle.setLaneChangeMode(veh_id, 0)
    traci.vehicle.setActionStepLength(veh_id,0.5)

# Define the Q-network model
# class QNetwork(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(QNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_size, 24)
#         self.fc2 = nn.Linear(24, 24)
#         self.fc3 = nn.Linear(24, action_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)




# Initialize the model, optimizer, and other parameters
state_space_size = 5  # id, pos, speed, acceleration, direction for each vehicle
action_space_size = 5  # changelane, accelerate, decelerate, changeRoute, maintain_speed
learning_rate = 0.001
discount_factor = 0.99
epsilon_decay = 0.995
min_epsilon = 0.01
epsilon = 0.8

# model = QNetwork(state_space_size, action_space_size)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


policy_model = PolicyNetwork(state_space_size, action_space_size)
optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
# Define the training parameters
total_episodes = 500
max_steps_per_episode = 50
gamma = 0.99



# Training loop
for episode in range(total_episodes):
    traci.load(["-c", "demo2.sumocfg", "--start", "--quit-on-end","--collision.stoptime", "10","--collision.action","None", "--time-to-teleport", "-2"])

    total_reward = 0
    v1 = int(random.uniform(8, 15))
    v2 = int(random.uniform(8, 15))
    episode_states = list()
    episode_actions = list()
    episode_rewards = list()
    for step in range(max_steps_per_episode):
        # Retrieve state information from the environment (SUMO)
        traci.simulationStep()
        loaded_vehicles = traci.vehicle.getLoadedIDList()
        if not traci.vehicle.getLoadedIDList():
            print("test")
            continue

        if step == v1:
            traci.vehicle.setSpeed("flow1.0", 0)
            traci.vehicle.setColor("flow1.0", (255,0,0))
            
        
        if step == v2:
            traci.vehicle.setSpeed("flow2.0", 0)
            traci.vehicle.setColor("flow2.0", (255,0,0))

        
        for veh_id in loaded_vehicles:
            if veh_id == "flow1.0" or veh_id == "flow2.0":
                continue
            if step < 5:
                add_aggressive_behavior(veh_id)

            state = [
                traci.vehicle.getPosition(veh_id)[0],
                traci.vehicle.getPosition(veh_id)[1],
                traci.vehicle.getSpeed(veh_id),
                traci.vehicle.getAcceleration(veh_id),
                traci.vehicle.getAngle(veh_id)
            ]

            state = torch.tensor(np.reshape(state, [1, state_space_size]), dtype=torch.float32)
            
            # epsilon = epsilon * (1.0 / (1.0 + epsilon_decay * episode))
            # q_values = model(state)
            
            action_probabilities = policy_model(state)
            action = torch.multinomial(action_probabilities, 1).item()

            # if np.random.rand() < epsilon:
            #     action = np.random.randint(len(q_values[0]))
            # else:
            #     action = torch.argmax(q_values).item()
            # Perform the chosen action using TraCI (update based on your specific actions)
            if action == 0:
                traci.vehicle.setSpeed(veh_id,state[0][2])
            elif action == 1:
                traci.vehicle.changeLane(veh_id, 1, duration=100)
            elif action == 2:
                traci.vehicle.setAccel(veh_id, 10)
            elif action == 3:
                traci.vehicle.setDecel(veh_id, 10)
            elif action == 4:
                if traci.vehicle.getRoadID(veh_id) == "E1":
                    traci.vehicle.setRoute(veh_id,["E1", "E2", "E3", "E5", "E6"])
                elif traci.vehicle.getRoadID(veh_id) == "E0":
                    traci.vehicle.setRoute(veh_id,["E0", "E1", "E2", "E3", "E5", "E6"])
            # Update the reward based on the simulation feedback
            collision_occurred = traci.simulation.getCollidingVehiclesIDList()
              # Implement this function based on your scenario
            
            if veh_id in collision_occurred:
                reward = -50
                # print("COLLISION!")
            elif traci.vehicle.getRoadID(veh_id) == "E6":
                # print("REACHED!")
                reward = 250
            elif traci.vehicle.getSpeed(veh_id) == 0:
                reward = -500
            else:
                reward = 1


            episode_states.append(state)
            episode_actions = list(episode_actions)
            episode_actions.append(action)
            episode_rewards.append(reward)

            # total_reward += reward
            # Update the Q-value using the Bellman equation 
            # 
            # next_state = [
            #     traci.vehicle.getPosition(veh_id)[0],
            #     traci.vehicle.getPosition(veh_id)[1],
            #     traci.vehicle.getSpeed(veh_id),
            #     traci.vehicle.getAcceleration(veh_id),
            #     traci.vehicle.getAngle(veh_id)
            # ]
            # next_state = torch.tensor(np.reshape(next_state, [1, state_space_size]), dtype=torch.float32)
            # print(state==next_state)
            # next_q_values = model(next_state)  # For simplicity, assume the next state is the same as the current state
            
            # target = reward + discount_factor * torch.max(next_q_values)
            # q_values[0][action] = (1 - discount_factor) * q_values[0][action] + discount_factor * target

            # Train the model
            # loss = nn.MSELoss()(model(state), q_values)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
        returns = []
        R = 0
        for r in reversed(episode_rewards):
            R = r + gamma * R
            returns.insert(0, R)

        # episode_actions = [a.item() for a in episode_actions]
        returns = torch.tensor(returns, dtype=torch.float32)
        episode_actions = torch.tensor(episode_actions, dtype=torch.long)

        log_probabilities = []
        for i, state in enumerate(episode_states):
            action_probabilities = policy_model(state)
            log_probabilities.append(torch.log(action_probabilities[0, episode_actions[i]]))

        if not log_probabilities:
            continue
        policy_loss = -torch.stack(log_probabilities) * returns
        optimizer.zero_grad()
        policy_loss.sum().backward()
        optimizer.step()
        if torch.isnan(policy_loss).any() or torch.isinf(policy_loss).any():
            continue

    # Print the total reward for the episode
    print(f"Episode {episode + 1}/{total_episodes}, Total Reward: {R}")

# Save the trained model
# torch.save(model.state_dict(), 'multi_vehicle_rl_model.pth')

# Close the TraCI connection
traci.close()
