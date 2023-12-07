import traci
import sumolib
import numpy as np

addedVehicles = []
allVehicles = []
accelerated = []
decelerated = []

def add_aggressive_behavior(veh_id):
    traci.vehicle.setAccel(veh_id, 100)
    traci.vehicle.setDecel(veh_id, 4)
    traci.vehicle.setEmergencyDecel(veh_id, 0.1)
    traci.vehicle.setApparentDecel(veh_id, 0.1)
    traci.vehicle.setTau(veh_id, 0.1)
    traci.vehicle.setMinGap(veh_id,0)
    traci.vehicle.setImperfection(veh_id, 0.1)
    traci.vehicle.setRoutingMode(veh_id, 0)
    traci.vehicle.setActionStepLength(veh_id,0.2)

def addAggressiveToAllVehicles():
    allVehicles = traci.vehicle.getLoadedIDList()
    for vehicle in allVehicles:
        if vehicle not in addedVehicles:
            add_aggressive_behavior(vehicle)
            contextSubscription(vehicle)
            addedVehicles.append(vehicle)

def contextSubscription(vehicle):
    desiredRange = 20
    traci.vehicle.subscribeContext(vehicle, traci.constants.CMD_GET_VEHICLE_VARIABLE, desiredRange)

def getV2VState(vehicle):
    return [getState(vehicle), getNearByVehicles(vehicle)]

def getState(vehicle):
    id = vehicle
    current_vehicles = traci.vehicle.getLoadedIDList()
    if vehicle in current_vehicles:
        pos = traci.vehicle.getPosition(vehicle)
        speed = traci.vehicle.getSpeed(vehicle)
        acc = traci.vehicle.getAccel(vehicle)
        angle = traci.vehicle.getAngle(vehicle)
        lane = traci.vehicle.getLaneID(vehicle)
        return np.array([id, pos[0], pos[1], speed, acc, angle, lane[-1]], dtype=str)
    else:
        
        return np.array([0, 0, 0, 0, 0, 0, 0])

def getNearByVehicles(vehicle):
    nearbyVehicles = np.array([])
    res = traci.vehicle.getContextSubscriptionResults(vehicle)
    for i in res:
        if i != vehicle:
            nearbyVehicles = np.append(nearbyVehicles, getState(i))
    return nearbyVehicles
    
def start_simulation():
    sumoBinary = sumolib.checkBinary("sumo-gui")
    sumoCmd = [sumoBinary, "-c", "demo2.sumocfg", "--start", "--collision.stoptime", "100", "--time-to-teleport", "-2"]
    traci.start(sumoCmd)

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

def perform_action(vehicle, action):
    if action[0]:
        traci.vehicle.setAccel(vehicle, traci.vehicle.getAccel(vehicle)+5)
        accelerated.append(vehicle)
    elif action[1]:
        traci.vehicle.setAccel(vehicle, traci.vehicle.getAccel(vehicle)-5)
        decelerated.append(vehicle)
    elif action[2]:
        traci.vehicle.setAccel(vehicle, 0)
    elif action[3]:
        traci.vehicle.changeLane(vehicle, 0 if getState(vehicle)[6] == 1 else 1, 300)
    elif action[4]:
        routes = traci.route.getIDList()
        if traci.vehicle.getRouteID(vehicle) == routes[0]:
            traci.vehicle.setRoute(vehicle, routes[1])
        else:
            traci.vehicle.setRoute(vehicle, routes[0])
    
def calculate_reward(vehicle):
    reward = 0
    collision = traci.simulation.getCollidingVehiclesIDList()
    current_vehicles = traci.vehicle.getLoadedIDList()
    if collision is not None and vehicle in collision:
        reward += rewards.collision_penalty
    if vehicle in allVehicles and vehicle not in current_vehicles:
        reward += rewards.end_reward
    if vehicle in accelerated or vehicle in decelerated:
        reward += rewards.accel_change_penalty
    else:
        reward += rewards.speed_reward
    return reward

# def state_space_to_array(statespace):
#     # print("ASDFGHJKL")
#     print(statespace)
#     # print("Satvika")
#     # print(curr)
#     curr = np.delete(curr, 0)
    
#     # print(curr)

rewards = []
    
# You'll need to replace the random state generation and action selection with actual V2V simulation data and logic.
def run_simulation(model, rewards):
    # traci.load(["-c","demo2.sumocfg","--start","--quit-on-end"])
    rewards = rewards
    step = 0
    while step < 30:
        if step == 8:
            traci.vehicle.setSpeed("flow1.0", 0)
            traci.vehicle.setSpeed("flow2.0", 0)
        currentVehicles = traci.vehicle.getLoadedIDList()
        # print("vehicles:", currentVehicles)
        state_space = {}
        for vehicle in currentVehicles:
            if vehicle not in allVehicles:
                allVehicles.append(vehicle)
            contextSubscription(vehicle)
            state_space[vehicle] = getV2VState(vehicle)
            #print("vehicle state space: ", vehicle , " space : ", state_space[vehicle])
        addAggressiveToAllVehicles()
        current_actions = {}
        # print("state space before for: ", state_space)
        for vehicle_state in state_space:
            #print("1")
            action = model.get_action(state_space[vehicle_state])
            #print(vehicle_state)
            current_actions[vehicle_state] = action
            contradictions = check_contradictions(action)
            if contradictions != -1:
                perform_action(vehicle_state, action)
        # print("state space after for: ", state_space)
        #print("2")
        traci.simulationStep()
        done = False
        #print("3")
        # print("state space after step: ", state_space)
        if step == 30:
            done = True
        for vehicle in currentVehicles:
            reward = calculate_reward(vehicle)
            # print("ghjk", state_space)
            # state_space_to_array(state_space)
            model.remember(state_space[vehicle], current_actions[vehicle], reward, getV2VState(vehicle), done)
        step += 1 
        # print("4")
    
        
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