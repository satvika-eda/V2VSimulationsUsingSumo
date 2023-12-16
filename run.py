from simulation import *
from DQNModelTorch import *
import torch.optim as optim

# Start the simulation
start_simulation()

# Training the model for 100 episodes
for i in range(100):
    # Run the simulation
    run_simulation(model)

    # Train the model using the memory with a batch-size of 50
    train_dqn(model, 50)

    # Clearing the memory
    model.memory = []

# Closing the simulation    
traci.close() 


