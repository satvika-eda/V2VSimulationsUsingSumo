import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# state params : Position, Velocity, Acceleration, direction
# action params : acc/decc, lane changing, turning 
# direction: right +ve, up +ve else -ve

### state representation
class V2VState:
    def __init__(self, ego_vehicle, nearby_vehicles):
        self.ego_vehicle = ego_vehicle
        self.nearby_vehicles = nearby_vehicles

class VehicleState:
    def __init__(self, position, speed, acceleration, direction):
        self.position = position
        self.speed = speed
        self.acceleration = acceleration
        self.direction = direction

##Sample data

ego_vehicle_state = VehicleState(position=(0, 0), speed=10, acceleration=0, direction=())
nearby_vehicle_states = [
    VehicleState(position=(5, 2), speed=15, acceleration=2, direction=()),
    VehicleState(position=(-3, 1), speed=12, acceleration=1, direction=())
]

ego_vehicle = V2VState(ego_vehicle_state, nearby_vehicle_states)


### action representation

class V2VActions:
    def __init__(self, accelerate, change_lane, turn):
        self.accelerate = accelerate
        self.change_lane = change_lane
        self.turn = turn

# Example Usage:
ego_vehicle_actions = V2VActions(accelerate=True, change_lane=True, turn=False)


### reward 

class V2VRewards:
    def __init__(self, collision_penalty, speed_reward, safety_reward, efficiency_reward):
        self.collision_penalty = collision_penalty
        self.speed_reward = speed_reward
        self.safety_reward = safety_reward
        self.efficiency_reward = efficiency_reward

# Example Usage:
rewards = V2VRewards(collision_penalty=-100, speed_reward=5, safety_reward=10, efficiency_reward=8)

# Functions to calculate rewards based on specific conditions
def calculate_reward(state, action, next_state):
    reward = 0
    # Check conditions and assign rewards based on actions and state transitions
    # For instance, penalize collisions
    if detect_collision(state, action, next_state):
        reward += rewards.collision_penalty
    else:
        # Reward for maintaining speed
        reward += rewards.speed_reward * calculate_speed_reward(state, next_state)
        # Reward for safe driving behavior
        reward += rewards.safety_reward * calculate_safety_reward(state, action, next_state)
        # Reward for efficient driving
        reward += rewards.efficiency_reward * calculate_efficiency_reward(state, action, next_state)
    return reward

# Implement functions to calculate specific rewards based on the state, action, and next_state
def detect_collision(state, action, next_state):
    # Implement collision detection logic
    return False  # Return True if collision is detected, False otherwise

def calculate_speed_reward(state, next_state):
    # Calculate reward based on maintaining speed or achieving desired speed
    return 1 if next_state.velocity >= state.velocity else -1

def calculate_safety_reward(state, action, next_state):
    # Calculate reward based on safe driving behavior
    # This could involve maintaining a safe distance, avoiding abrupt maneuvers, etc.
    return 1 if is_safe_action(action) else -1

def calculate_efficiency_reward(state, action, next_state):
    # Calculate reward for efficient driving
    # This could involve taking actions that lead to reaching the destination faster, etc.
    return 1 if is_efficient_action(action) else -1

# Helper functions to determine safe and efficient actions (to be implemented based on specific criteria)
def is_safe_action(action):
    # Implement logic to determine if the action is safe
    return True

def is_efficient_action(action):
    # Implement logic to determine if the action is efficient
    return True


### DQN 

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.array(random.sample(self.memory, batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, self.state_size))[0])
            target_f = self.model.predict(state.reshape(1, self.state_size))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, self.state_size), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define state and action sizes
state_size = 10  # Adjust based on your actual state representation size
action_size = 5   # Adjust based on the number of actions available
state_set = [ego_vehicle]
# Initialize the DQN agent
agent = DQNAgent(state_size, action_size)

# Training loop (Replace this with your V2V simulation environment interactions)
for episode in range(NUM_EPISODES):
    for ego_v in state_set:
        state = ego_v
        done = False
        while not done:
            action = agent.act(state)
            next_state = np.random.random((1, state_size))  # Replace this with your actual next state
            reward = calculate_reward(state, action, next_state)  # Calculate reward based on your reward function
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(batch_size=32)  # Train the agent

# Define your reward calculation function based on your V2V simulation context
def calculate_reward(state, action, next_state):
    # Implement your reward logic based on the V2V simulation states, actions, and transitions
    return 0  # Replace this with your calculated reward

# You'll need to replace the random state generation and action selection with actual V2V simulation data and logic.
