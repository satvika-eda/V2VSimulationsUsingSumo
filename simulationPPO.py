import traci
import sumolib
import math
import random
import numpy as np
import torch


addedVehicles = []
allVehicles = []
accelerated = []
decelerated = []
stopped_vehicles = ['flow1.0', 'flow2.0']

def add_aggressive_behavior(veh_id):
    traci.vehicle.setAccel(veh_id, 100)
    # traci.vehicle.setDecel(veh_id, 4)
    traci.vehicle.setEmergencyDecel(veh_id, 0.1)
    traci.vehicle.setApparentDecel(veh_id, 0.1)
    traci.vehicle.setTau(veh_id, 0.1)
    # traci.vehicle.setMinGap(veh_id,0)
    traci.vehicle.setImperfection(veh_id, 0.1)
    traci.vehicle.setLaneChangeMode(veh_id, 0)
    # traci.vehicle.setRoutingMode(veh_id, 0)
    traci.vehicle.setActionStepLength(veh_id,0.2)

def addAggressiveToAllVehicles():
    allVehicles = traci.vehicle.getLoadedIDList()
    for vehicle in allVehicles:
        if vehicle not in addedVehicles:
            add_aggressive_behavior(vehicle)
            contextSubscription(vehicle)
            addedVehicles.append(vehicle)

def contextSubscription(vehicle):
    if vehicle not in stopped_vehicles:
        desiredRange = 30
        traci.vehicle.subscribeContext(vehicle, traci.constants.CMD_GET_VEHICLE_VARIABLE, desiredRange)

def getV2VState(vehicle):
    if vehicle not in stopped_vehicles:
        # print("cvbn", type(getState(vehicle)[0]))
        # print("n v : ", getNearByVehicles(vehicle))
        # print("cvbn2", type(getNearByVehicles(vehicle)[0]))
        state = np.concatenate((getState(vehicle), getNearByVehicles(vehicle)), dtype=np.float32)
        while len(state) != 60:
            state = np.append(state, np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))
        return state
    # return np.array([getState(vehicle), getNearByVehicles(vehicle)], dtype=np.float32)

def getState(vehicle):
    if vehicle not in stopped_vehicles:
    # id = vehicle
        current_vehicles = traci.vehicle.getLoadedIDList()
        if vehicle in current_vehicles:
            #print("vehicle: ",vehicle)
            pos = traci.vehicle.getPosition(vehicle) 
        
            #print("pos : ", pos)
            speed = traci.vehicle.getSpeed(vehicle)
            if math.isnan(speed):
                speed = 0

            #print("speed : ", speed)
            acc = traci.vehicle.getAccel(vehicle)
            #print("acc : ", acc)
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
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

def getNearByVehicles(vehicle):
    if vehicle not in stopped_vehicles:
        nearbyVehicles = np.array([], dtype=np.float32)
        res = traci.vehicle.getContextSubscriptionResults(vehicle)
        #("context", res)
        for i in res:
            if i != vehicle:
                nearbyVehicles = np.append(nearbyVehicles, getState(i))
        #return nearbyVehicles
        arr = nearbyVehicles[nearbyVehicles != None]
        arr = np.array(arr, dtype=np.float32)
        return arr
    else:
        return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    
def start_simulation():
    sumoBinary = sumolib.checkBinary("sumo-gui")
    sumoCmd = [sumoBinary, "-c", "demo2.sumocfg", "--start", "--collision.stoptime", "100", "--time-to-teleport", "-2"]
    #sumoCmd = [sumoBinary, "-c", "demo2.sumocfg"]
    traci.start(sumoCmd)

def check_contradictions(action):
    # if action[0] and action[1] and action[2]:
    #     return -1
    # elif action[3] and action[4]:
    #     return -1
    if action[0] and action[1]:
        return -1
    elif action[0] and action[2]:
        return -1
    elif action[1] and action[2]:
        return -1
    elif not action[0] and not action[1] and not action[2] and not action[3] and not action[4]:
        return -1
    else:
        return 0

def perform_action(vehicle, action):
    routes1 = {'E0': ('E0', 'E1', 'E2', 'E3', 'E5', 'E6'), 'E1': ('E1', 'E2', 'E3', 'E5', 'E6'), 'E2': ('E2', 'E3', 'E5', 'E6'), 'E3': ('E3', 'E5', 'E6'), 'E4': ('E4', 'E6'), 'E5': ('E5', 'E6'), 'E6': ('E6'), 'J2_0': ('E2', 'E3', 'E5', 'E6'), 'J2_1': ('E2', 'E3', 'E5', 'E6')}
    routes2 = {'E0': ('E0', 'E1', 'E4', 'E6'), 'E1': ('E1', 'E4', 'E6'), 'E2': ('E2', 'E3', 'E5', 'E6'), 'E3': ('E3', 'E5', 'E6'), 'E4': ('E4', 'E6'), 'E5': ('E5', 'E6'), 'E6': ('E6'), 'J2_0': ('E4', 'E6'), 'J2_1': ('E4', 'E6')}
    if vehicle not in stopped_vehicles:
        if vehicle in traci.vehicle.getLoadedIDList():
            if action[0]:
                traci.vehicle.setAccel(vehicle, traci.vehicle.getAccel(vehicle)+5)
                accelerated.append(vehicle)
            elif action[1]:
                a = traci.vehicle.getAccel(vehicle)
                # print("accelration: ", a)
                # print("acc type: ", type(a))
                acc = traci.vehicle.getAccel(vehicle) - 5
                if acc > 0:
                    traci.vehicle.setAccel(vehicle, acc)
                else:
                    traci.vehicle.setDecel(vehicle, math.fabs(acc))
                decelerated.append(vehicle)
            elif action[2]:
                traci.vehicle.setAccel(vehicle, 0)
            elif action[3]:
                print("I came here")
                traci.vehicle.changeLane(vehicle, 0 if getState(vehicle)[5] == 1 else 1, 300)
            elif action[4]:
                print("I'm here")
                road = traci.vehicle.getRoadID(vehicle)
                # if traci.vehicle.getRoadID(vehicle) == routes[0]:
                #     traci.vehicle.setRoute(vehicle, traci.route.getEdges("r_1"))
                # else:
                #     traci.vehicle.setRoute(vehicle, traci.route.getEdges("r_0"))
                route = random.randint(0, 1)
                if route == 0:
                    if road in routes1:
                        traci.vehicle.setRoute(vehicle, routes1[road])
                    elif road in routes2:
                        traci.vehicle.setRoute(vehicle, routes2[road])
                
    
def calculate_reward(vehicle):
    reward = 0
    if vehicle not in stopped_vehicles:
        if vehicle in traci.vehicle.getLoadedIDList():
            collision = traci.simulation.getCollidingVehiclesIDList()
            current_vehicles = traci.vehicle.getLoadedIDList()
            if collision is not None and vehicle in collision:
                reward += rewards.collision_penalty
            if traci.vehicle.getRoadID(vehicle) == 'E6':
                reward += rewards.end_reward
            if traci.vehicle.getSpeed(vehicle) == 0:
                reward += rewards.stop_penalty
            # if vehicle in accelerated or vehicle in decelerated:
            #     reward += rewards.accel_change_penalty
            # else:
            #     reward += rewards.speed_reward
        return reward


rewards = []
    
# You'll need to replace the random state generation and action selection with actual V2V simulation data and logic.
def run_simulation(model, rewards):
    #traci.load(["-c","demo2.sumocfg","--start","--quit-on-end"])
    traci.load(["-c", "demo2.sumocfg", "--start", "--quit-on-end", "--collision.stoptime", "100","--collision.action", "None", "--time-to-teleport", "-2"])
    # time.sleep(2)
    rewards = rewards
    step = 0
    s1 = random.randint(5,15)
    s2 = random.randint(8,15)
    new_states = []
    new_actions = []
    new_rewards = []
    log_probs = []
    while step < 100:
        print("step: ", step)
        if step == s1:
            traci.vehicle.setSpeed("flow1.0", 0)
            traci.vehicle.setLaneChangeMode("flow1.0",0)
        if step == s2:
            traci.vehicle.setSpeed("flow2.0", 0)
            traci.vehicle.setLaneChangeMode("flow2.0",0)
        currentVehicles = traci.vehicle.getLoadedIDList()
        # print("vehicles:", currentVehicles)
        state_space = {}
        for vehicle in currentVehicles:
            if vehicle not in stopped_vehicles:
                if vehicle not in allVehicles:
                    allVehicles.append(vehicle)
                contextSubscription(vehicle)
                state_space[vehicle] = getV2VState(vehicle)
            #print("POIs: ", state_space[vehicle])
            #print("vehicle state space: ", vehicle , " space : ", state_space[vehicle])
        addAggressiveToAllVehicles()
        current_actions = {}
        # print("state space before for: ", state_space)
        for vehicle_state in state_space:
            #print("1")
            # print("POI: ", state_space[vehicle_state])
            action, log_prob = model.get_action(state_space[vehicle_state])
            #print(vehicle_state)
            current_actions[vehicle_state] = action
            contradictions = check_contradictions(action)
            if contradictions != -1:
                perform_action(vehicle_state, action)
            else:
                indices = []
                for i in range(len(action)):
                    if action[i]:
                        indices.append(i)
                if len(indices) != 0:
                    index = random.randint(0, len(indices)-1)
                    actionTaken = [False, False, False, False, False]
                    actionTaken[indices[index]] = True
                    perform_action(vehicle_state, actionTaken)


        # print("state space after for: ", state_space)
        #print("2")
        traci.simulationStep()
        done = False
        #print("3")
        # print("state space after step: ", state_space)
        if step == 30:
            done = True
        for vehicle in currentVehicles:
            if vehicle not in stopped_vehicles:
                reward = calculate_reward(vehicle)
            # print("ghjk", state_space)
            # state_space_to_array(state_space)
                vehiclestate = getV2VState(vehicle)
                new_states.append(vehiclestate)
                new_actions.append(action)
                log_probs.append(log_prob)
                new_rewards.append(reward)
            # print(vehiclestate.shape)
            # print(vehiclestate)
        log_probs = torch.tensor(log_probs,dtype=torch.float32)
        model.update_policy(new_states, new_actions,log_probs, new_rewards)
        step += 1 
        # print("4")
    print("current episode ended")
    return 