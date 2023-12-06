from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2
import numpy as np
import random

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

class DQN():
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.action_size = action_size
        self.model = self.build_network()

    def build_network(self):
        network = Sequential()
        network.add(Dense(24, input_dim = self.state_size, activation='relu'))
        network.add(Dense(24, activation='relu'))
        network.add(Dense(self.action_size, activation='sigmoid'))
        network.compile(loss='mse', optimizer=adam_v2.Adam(lr=self.learning_rate))
        return network
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        action = []
        for i in act_values:
            if i > 0.5:
                action.append(True)
            else:
                action.append(False)
        return V2VActions(*action)
    
    def replay(self, batch_size):
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

    






