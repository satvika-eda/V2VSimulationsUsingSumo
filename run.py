from simulation import *
from DQNModelTorch import *
import torch.optim as optim

# Start the simulation
start_simulation()
epsilon = 1.0  # Initial epsilon value
epsilon_min = 0.01  # Minimum epsilon value
epsilon_decay = 0.995
# Training the model for 100 episodes
for i in range(500):
    # Run the simulation
    print("episode : ", i)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    run_simulation(model, epsilon) 
    # Train the model using the memory with a batch-size of 50
    train_dqn(model, 50)
    # Clearing the memory
    model.memory = []
graph()
# Closing the simulation    
traci.close()

