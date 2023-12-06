from simulation import *
from DQNModel import *
import torch.optim as optim

num_episodes = 10


state_size = 10
action_size = 5
learning_rate = 0.01

model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
replay_buffer = ReplayBuffer(capacity=10000)


for i in range(num_episodes):
    start_simulation()
    run_simulation(model)


