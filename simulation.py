import traci
import time
import sumolib
from DQNModelTorch import *
from DQNModelTorch import V2VRewards
import math
import random
import matplotlib.pyplot as plt
import numpy as np

addedVehicles = []
allVehicles = []
accelerated = []
decelerated = []
collidedVehicles = []
stopped_vehicles = ['flow1.0', 'flow2.0']

def add_aggressive_behavior(veh_id):
    traci.vehicle.setAccel(veh_id, 100)
    #traci.vehicle.setDecel(veh_id, 0)
    traci.vehicle.setEmergencyDecel(veh_id, 2)
    traci.vehicle.setApparentDecel(veh_id, 0.1)
    traci.vehicle.setTau(veh_id, 0.1)
    # traci.vehicle.setMinGap(veh_id,0)
    traci.vehicle.setImperfection(veh_id, 0.1)
    traci.vehicle.setLaneChangeMode(veh_id, 0)
    # traci.vehicle.setRoutingMode(veh_id, 0)
    traci.vehicle.setActionStepLength(veh_id, 0)

def addAggressiveToAllVehicles():
    allVehicles = traci.vehicle.getLoadedIDList()
    for vehicle in allVehicles:
        if vehicle not in addedVehicles:
            add_aggressive_behavior(vehicle)
            contextSubscription(vehicle)
            addedVehicles.append(vehicle)

def contextSubscription(vehicle):
    if vehicle not in stopped_vehicles:
        desiredRange = 50
        traci.vehicle.subscribeContext(vehicle, traci.constants.CMD_GET_VEHICLE_VARIABLE, desiredRange)

def getV2VState(vehicle):
    if vehicle not in stopped_vehicles:
        # print("cvbn", type(getState(vehicle)[0]))
        # print("n v : ", getNearByVehicles(vehicle))
        # print("cvbn2", type(getNearByVehicles(vehicle)[0]))
        state = np.concatenate((getState(vehicle), getNearByVehicles(vehicle)), dtype=np.float32)
        while len(state) != 7*2*6:
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
    sumoCmd = [sumoBinary, "-c", "demo2.sumocfg", "--start", "--collision.stoptime", "1", "--time-to-teleport", "-2"]
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
    # routes1 = {'E6': ('E6', 'E7', 'E8'), 'E7': ('E7', 'E8'), 'E8': ('E8')}
    # routes2 = {'E6': ('E6', 'E9', 'E10', 'E11', 'E8'), 'E9': ('E9', 'E10', 'E11', 'E8'), 'E10': ('E10', 'E11', 'E8'), 'E11': ('E11', 'E8'), 'E8': ('E8')}
    if vehicle not in stopped_vehicles:
        if vehicle in traci.vehicle.getLoadedIDList():
            if action[0]:
                traci.vehicle.setAccel(vehicle, traci.vehicle.getAccel(vehicle)+2)
                accelerated.append(vehicle)
            elif action[1]:
                a = traci.vehicle.getAccel(vehicle)
                # print("accelration: ", a)
                # print("acc type: ", type(a))
                acc = traci.vehicle.getAccel(vehicle) - 2
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
                print("I'm here--------------------")
                road = traci.vehicle.getRoadID(vehicle)
                route = traci.vehicle.getRoute(vehicle)
                print("Route: ", route)
                # if traci.vehicle.getRoadID(vehicle) == routes[0]:
                #     traci.vehicle.setRoute(vehicle, traci.route.getEdges("r_1"))
                # else:
                #     traci.vehicle.setRoute(vehicle, traci.route.getEdges("r_0"))
                # route = random.randint(0, 1)
                if len(route) != 6:
                    if road in routes1:
                        traci.vehicle.setRoute(vehicle, routes1[road])
                else:        
                    if road in routes2:
                        traci.vehicle.setRoute(vehicle, routes2[road])
            print("action taken")

rewards = V2VRewards(-400, 1000, -80, -300)               
    
def calculate_reward(vehicle):
    reward = 0
    if vehicle not in stopped_vehicles:
        if vehicle in traci.vehicle.getLoadedIDList():
            collision = traci.simulation.getCollidingVehiclesIDList()
            # print("Collision: ", collision)
            edge = traci.vehicle.getRoadID(vehicle)
            
            if vehicle not in traci.vehicle.getIDList():
                print(vehicle)
                reward += rewards.end_reward
                print("reached end **********************************")
            if collision is not None and vehicle in collision:
                reward += rewards.collision_penalty
                traci.vehicle.setSpeed(vehicle, 0)
                traci.vehicle.setEmergencyDecel(vehicle,100)
                traci.vehicle.setColor(vehicle, (255, 0, 0))
                collidedVehicles.append(vehicle)
                #traci.vehicle.remove(vehicle)
                # print("Collision: ", collision)
                # print("Collision penalty added ---- ")
                return reward
            if traci.vehicle.getSpeed(vehicle) == 0 and vehicle not in collision:
                reward += rewards.stop_penalty
            # if vehicle in accelerated or vehicle in decelerated:
            #     reward += rewards.accel_change_penalty
            # else:
            #     reward += rewards.speed_reward
        print("calcu;ated reward")
        return reward

# def state_space_to_array(statespace):
#     # print("ASDFGHJKL")
#     print(statespace)
#     # print("Satvika")
#     # print(curr)
#     curr = np.delete(curr, 0)
    
#     # print(curr)

def random_action():
    action = np.array([])
    for i in range(5):
        select = np.random.rand()
        if select >= 0.5:
            action = np.append(action, [True])
        else:
            action = np.append(action, [False])
    return action

lst = []

# You'll need to replace the random state generation and action selection with actual V2V simulation data and logic.
def run_simulation(model, epsilon):
    #traci.load(["-c","demo2.sumocfg","--start","--quit-on-end"])
    traci.load(["-c", "demo2.sumocfg", "--start", "--quit-on-end", "--collision.stoptime", "100", "--time-to-teleport", "-2"])
    # time.sleep(2)
    step = 0
    finalR = 0
    s1 = 8
    s2 = 15
    # s3 = random.randint(20,35)
    while step < 100:
        #print("step: ", step)
        if step == s1:
            traci.vehicle.setSpeed("flow1.0", 0)
            traci.vehicle.setLaneChangeMode("flow1.0",0)
            traci.vehicle.setColor("flow1.0", (255, 0, 0))
        if step == s2:
            traci.vehicle.setSpeed("flow2.0", 0)
            traci.vehicle.setLaneChangeMode("flow2.0",0)
            traci.vehicle.setColor("flow2.0", (255, 0, 0))
        # if step == s3:
        #     traci.vehicle.setSpeed("flow1.1", 0)
        #     traci.vehicle.setLaneChangeMode("flow1.1",0)
        #     traci.vehicle.setColor("flow1.1", (255, 0, 0))
        # if step == s3:
        #     traci.vehicle.setSpeed("flow3.0", 0)
        #     traci.vehicle.setLaneChangeMode("flow3.0",0)
        #     traci.vehicle.setColor("flow3.0", (255, 0, 0))
        currentVehicles = traci.vehicle.getLoadedIDList()
        # print("vehicles:", currentVehicles)
        state_space = {}
        for vehicle in currentVehicles:
            if vehicle not in stopped_vehicles:
                if vehicle not in allVehicles:
                    allVehicles.append(vehicle)
                contextSubscription(vehicle)
                if vehicle not in collidedVehicles:
                    state_space[vehicle] = getV2VState(vehicle)
            #print("POIs: ", state_space[vehicle])
            #print("vehicle state space: ", vehicle , " space : ", state_space[vehicle])
        addAggressiveToAllVehicles()
        current_actions = {}
        # print("state space before for: ", state_space)
        for vehicle_state in state_space:
            #print("1")
            # print("POI: ", state_space[vehicle_state])
            # if np.random.rand() > epsilon:
            #     print("predict action")
            action = model.predict_action(state_space[vehicle_state])
            # else:
            #     action = random_action()
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
                    print("action taken : ", actionTaken)
                    perform_action(vehicle_state, actionTaken)


        # print("state space after for: ", state_space)
        #print("2")

        traci.simulationStep()

        done = False
        #print("3")
        # print("state space after step: ", state_space)
        if step == 100:
            done = True
        for vehicle in state_space:
            if vehicle not in stopped_vehicles:
                reward = calculate_reward(vehicle)
                print("vehicle:", vehicle)
                print("reward:", reward)
                # print("vehicle : ", vehicle, " reward: ", reward)
            # print("ghjk", state_space)
            # state_space_to_array(state_space)
                # print("calcu;ated reward run")
                # vehiclestate = getV2VState(vehicle)
                # print("v2v get done")
            # print(vehiclestate.shape)
            # print(vehiclestate)
                model.remember(state_space[vehicle], current_actions[vehicle], getV2VState(vehicle), reward, done)
                if vehicle=="flow1.1":
                    print(traci.vehicle.getRoadID(vehicle), " 1.2 in edge " )
                    finalR += reward
                    print("1.2 : ", reward)
        step += 1 
        # print("4")
    lst.append(finalR)
    print("episode rward ", finalR)
    print("current episode ended")
        
        # if step == 8:
        #     traci.vehicle.setSpeed("flow1.0", 0)
        #     traci.vehicle.setColor("flow1.0", (255,0,0))
        #     traci.vehicle.setSpeed("flow2.0", 0)
        #     traci.vehicle.setColor("flow2.0", (255,0,0))
        # collisions = traci.simulation.getCollisions()
        # for collision in collisions:
        #     traci.vehicle.setSpeed(collision.collider, 0)
        #     traci.vehicle.setEmergencyDecel(collision.collider, 1000)
        #     traci.vehicle.setColor(collision.collider, (255,0,0))
        #     traci.vehicle.setSpeed(collision.victim, 0)

        
# def graph():
#     print(lst)
#     plt.plot(lst)
#     plt.savefig("abc1.png")