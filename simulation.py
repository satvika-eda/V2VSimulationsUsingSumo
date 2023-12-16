import traci
import time
import sumolib
from DQNModelTorch import *
import math
import numpy as np
import random 

addedVehicles = []
allVehicles = []
accelerated = []
decelerated = []



# Defining aggressive behaviour to promote more collisions in the simulation, as sumo takes care of the aggressive behavior in general.
def add_aggressive_behavior(veh_id):
    traci.vehicle.setAccel(veh_id, 100)
    traci.vehicle.setDecel(veh_id, 4)
    traci.vehicle.setEmergencyDecel(veh_id, 0.1)
    traci.vehicle.setApparentDecel(veh_id, 0.1)
    traci.vehicle.setTau(veh_id, 0.1)
    traci.vehicle.setImperfection(veh_id, 0.1)
    traci.vehicle.setLaneChangeMode(veh_id, 0)
    traci.vehicle.setActionStepLength(veh_id,0.2)


# function to add the defined agressive behaviour to all vehicles.
def addAggressiveToAllVehicles():
    allVehicles = traci.vehicle.getLoadedIDList()
    for vehicle in allVehicles:
        if vehicle not in addedVehicles:
            add_aggressive_behavior(vehicle)
            contextSubscription(vehicle)
            addedVehicles.append(vehicle)

# Helper function to get details of nearby cars from SUMO
def contextSubscription(vehicle):
    desiredRange = 30
    traci.vehicle.subscribeContext(vehicle, traci.constants.CMD_GET_VEHICLE_VARIABLE, desiredRange)

# Getting the state of a given vehicle and its nearby vehicle
def getV2VState(vehicle):
    state = np.concatenate((getState(vehicle), getNearByVehicles(vehicle)), dtype=np.float32)
    while len(state) != 60:
        state = np.append(state, np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))
    return state


# Getting the state information from SUMO using traci
def getState(vehicle):
    current_vehicles = traci.vehicle.getLoadedIDList()
    if vehicle in current_vehicles:
        pos = traci.vehicle.getPosition(vehicle) 
    
        speed = traci.vehicle.getSpeed(vehicle)
        acc = traci.vehicle.getAccel(vehicle)
        angle = traci.vehicle.getAngle(vehicle)
        if angle in range(-360,360):
            lane = traci.vehicle.getRoadID(vehicle)
            pos = (0,0)
            speed = 10.0
            angle = 90.0
        else:
            if vehicle[-3] == 1:
                lane = "0"
            else:
                lane = "1"
        

        return np.array([pos[0], pos[1], speed, acc, angle, lane[-1]], dtype=np.float32)
    else:
        return np.array([0, 0, 0, 0, 0, 0])


# Getting states of nearby vehicles in SUMO using traci.
def getNearByVehicles(vehicle):
    nearbyVehicles = np.array([])
    res = traci.vehicle.getContextSubscriptionResults(vehicle)
    for i in res:
        if i != vehicle:
            nearbyVehicles = np.append(nearbyVehicles, getState(i))
    return nearbyVehicles
    

# Starting Simulation
def start_simulation():
    sumoBinary = sumolib.checkBinary("sumo-gui")
    sumoCmd = [sumoBinary, "-c", "demo2.sumocfg", "--start", "--collision.stoptime", "100", "--time-to-teleport", "-2"]
    traci.start(sumoCmd)


# Checking for contradicting actions.
def check_contradictions(action):
    if action[0] and action[1] and action[2]:
        return -1
    elif action[3] and action[4]:
        return -1
    elif action[0] and action[1]:
        return -1
    elif action[0] and action[2]:
        return -1
    elif action[1] and action[2]:
        return -1
    else:
        return 0


# Performing an action on the given vehicle.
def perform_action(vehicle, action):
    routes = {'E0': ('E0', 'E1', 'E2', 'E3', 'E5', 'E6'), 'E1': ('E1', 'E2', 'E3', 'E5', 'E6'), 'E2': ('E2', 'E3', 'E5', 'E6'), 'E3': ('E3', 'E5', 'E6'), 'E4': ('E4', 'E6'), 'E5': ('E5', 'E6'), 'E6': ('E6')}
    if vehicle != 'flow1.0' or vehicle != 'flow2.0':
        if vehicle in traci.vehicle.getLoadedIDList():
            if action[0]:
                #Action - Acceleration
                traci.vehicle.setAccel(vehicle, traci.vehicle.getAccel(vehicle)+5)
                accelerated.append(vehicle)
            elif action[1]:
                #Action - Deceleration
                a = traci.vehicle.getAccel(vehicle)
                acc = traci.vehicle.getAccel(vehicle) - 5
                if acc > 0:
                    traci.vehicle.setAccel(vehicle, acc)
                else:
                    traci.vehicle.setDecel(vehicle, math.fabs(acc))
                decelerated.append(vehicle)
            elif action[2]:
                #Action - Maintain Speed
                traci.vehicle.setAccel(vehicle, 0)
            elif action[3]:
                #Action - Change Lane
                traci.vehicle.changeLane(vehicle, 0 if getState(vehicle)[5] == 1 else 1, 300)
            elif action[4]:
                #Action - Change Route
                road = traci.vehicle.getRoadID(vehicle)
                if road in routes:
                    traci.vehicle.setRoute(vehicle, routes[road])
    
# Calculating reward based on the state of the vehicle.
def calculate_reward(vehicle):
    reward = 0
    if vehicle != 'flow1.0' or vehicle != 'flow2.0':
        if vehicle in traci.vehicle.getLoadedIDList():
            collision = traci.simulation.getCollidingVehiclesIDList()
            current_vehicles = traci.vehicle.getLoadedIDList()
            if collision is not None and vehicle in collision:
                reward += rewards.collision_penalty
            if traci.vehicle.getRoadID(vehicle) == 'E6':
                reward += rewards.end_reward
            if traci.vehicle.getSpeed(vehicle) == 0:
                reward += rewards.stop_penalty
        return reward


# Function to Run the simulation for one episode
def run_simulation(model):
    traci.load(["-c", "demo2.sumocfg", "--start", "--quit-on-end", "--collision.stoptime", "100","--collision.action", "None", "--time-to-teleport", "-2"])
    step = 0
    s1 = random.randint(10,15)
    s2 = random.randint(10,15)
    while step < 100:
        print("step: ", step)
        #Stopping two cars at random positions.
        if step == s1:
            traci.vehicle.setSpeed("flow1.0", 0)
            traci.vehicle.setLaneChangeMode("flow1.0",0)
        if step == s2:
            traci.vehicle.setSpeed("flow2.0", 0)
        currentVehicles = traci.vehicle.getLoadedIDList()
        state_space = {}
        for vehicle in currentVehicles:
            if vehicle != 'flow1.0' or vehicle != 'flow2.0':
                if vehicle not in allVehicles:
                    allVehicles.append(vehicle)
                contextSubscription(vehicle)
                state_space[vehicle] = getV2VState(vehicle)
        addAggressiveToAllVehicles()
        current_actions = {}

        #Performing actions
        for vehicle_state in state_space:
            action = model.predict_action(state_space[vehicle_state])
            current_actions[vehicle_state] = action
            contradictions = check_contradictions(action)
            if contradictions != -1:
                perform_action(vehicle_state, action)

        #Simulation steps
        traci.simulationStep()
        done = False
        if step == 30:
            done = True
        
        #Calculating rewards
        for vehicle in currentVehicles:
            reward = calculate_reward(vehicle)
            vehiclestate = getV2VState(vehicle)
            model.remember(state_space[vehicle], current_actions[vehicle], getV2VState(vehicle), reward, done)
        step += 1 
    print("current episode ended")


        