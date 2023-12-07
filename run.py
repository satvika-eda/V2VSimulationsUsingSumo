from simulation import *
from DQNModelTorch import *
import torch.optim as optim

start_simulation()
for i in range(100):
    print("episode : ", i)
    # print("simulation started")
    run_simulation(model) 
    # print("simulation ran as well") 
    train_dqn(model, 50)
    model.memory = []
traci.close() 


