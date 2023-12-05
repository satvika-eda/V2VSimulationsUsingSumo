import traci
import time
import sumolib

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

def addAggressiveToAllVehicles(addedVehicles):
    allVehicles = traci.vehicle.getLoadedIDList()
    for vehicle in allVehicles:
        if vehicle not in addedVehicles:
            add_aggressive_behavior(vehicle)
            contextSubscription(vehicle)
            addedVehicles.append(vehicle)
    return addedVehicles

def contextSubscription(vehicle):
    desiredRange = 10
    traci.vehicle.subscribeContext(vehicle, traci.constants.CMD_GET_VEHICLE_VARIABLE, desiredRange, [traci.constants.VAR_SPEED, traci.constants.VAR_POSITION, traci.constants.VAR_ANGLE, traci.constants.VAR_ACCELERATION])

# You'll need to replace the random state generation and action selection with actual V2V simulation data and logic.
def runSimulation():
    addedVehicles = []
    sumoBinary = sumolib.checkBinary("sumo-gui")
    sumoCmd = [sumoBinary, "-c", "demo2.sumocfg", "--start", "--collision.stoptime", "100", "--time-to-teleport", "-2", "--quit-on-end"]
    traci.start(sumoCmd)
    step = 0
    while step < 30:
        traci.simulationStep()
        time.sleep(1)
        addedVehicles = addAggressiveToAllVehicles(addedVehicles)
        if step == 8:
            traci.vehicle.setSpeed("flow1.0", 0)
            traci.vehicle.setColor("flow1.0", (255,0,0))
            traci.vehicle.setSpeed("flow2.0", 0)
            traci.vehicle.setColor("flow2.0", (255,0,0))
        collisions = traci.simulation.getCollisions()
        for collision in collisions:
            traci.vehicle.setSpeed(collision.collider, 0)
            traci.vehicle.setEmergencyDecel(collision.collider, 1000)
            traci.vehicle.setColor(collision.collider, (255,0,0))
            traci.vehicle.setSpeed(collision.victim, 0)
        step += 1

runSimulation() 

        
