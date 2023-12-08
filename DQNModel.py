import tensorflow as tf
from tensorflow.keras.models import Sequential

class V2VState:
    def __init__(self, ego_vehicle, nearby_vehicles):
        self.ego_vehicle = ego_vehicle
        self.nearby_vehicles = nearby_vehicles

class VehicleState:
    def __init__(self, id, position, speed, acceleration, direction):
        self.id = id
        self.position = position
        self.speed = speed
        self.acceleration = acceleration
        self.direction = direction


### action representation
class V2VActions:
    def __init__(self, accelerate, change_lane, turn):
        self.accelerate = accelerate
        self.change_lane = change_lane
        self.turn = turn

# Example Usage:
# ego_vehicle_actions = V2VActions(accelerate=True, change_lane=True, turn=False)


### reward 

class V2VRewards:
    def __init__(self, collision_penalty, speed_reward, safety_reward):
        self.collision_penalty = collision_penalty
        self.speed_reward = speed_reward
        self.safety_reward = safety_reward
        # self.efficiency_reward = efficiency_reward

# Example Usage:
rewards = V2VRewards(collision_penalty=-100, speed_reward=5, safety_reward=10)


class DQN():
    def __init__(self):
        pass

    def build_network():
        network = Sequential()




