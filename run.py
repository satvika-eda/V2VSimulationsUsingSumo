from simulation import *
from DQNModelTorch import *
import torch.optim as optim

start_simulation()
epsilon = 1.0  # Initial epsilon value
epsilon_min = 0.01  # Minimum epsilon value
epsilon_decay = 0.995 
for i in range(500):
    print("episode : ", i)
    # print("simulation started")
    # print("simulation ran as well") 
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    run_simulation(model, epsilon) 
    train_dqn(model, 50)
    model.memory = []
traci.close() 


