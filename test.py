import traci
import sumolib
import time
sumoBinary = sumolib.checkBinary("sumo-gui")

sumoCmd = [sumoBinary, "-c", "demo2.sumocfg", "--start", "--collision.stoptime", "100", "--time-to-teleport", "-2", "--quit-on-end"]
# "--collision.action", "none",
# "--collision-mingap-factor 0", "--collision.action warn"
traci.start(sumoCmd)
step = 0
# traci.simulation.setParameter("sim", "collision.action", "none")
# traci.simulation.setParameter("sim", "step-length", "0.2")
vehicles = traci.vehicle.getIDList()
print(vehicles)

subscriptionID = 0
# pos = traci.vehicle.getPosition(vehicles[0])
# edgeID = traci._vehicle.VehicleDomain.getRoadID(vehicles[0])
dist = 100  # Distance from the position along the edge
varIden = [vehicles]
vehicleID = "t_1"
desiredRange = 10
for vehicle in vehicles:
    traci.vehicle.subscribeContext(vehicle, traci.constants.CMD_GET_VEHICLE_VARIABLE, desiredRange, [traci.constants.VAR_SPEED, traci.constants.VAR_POSITION, traci.constants.VAR_ANGLE, traci.constants.VAR_ACCELERATION])

# traci.vehicle.subscribeContext(subscriptionID, pos, "E2_0", 10, varIden)
# traci.vehicle.subscribeContext(subscriptionID, 0, 10, varIDs=None, begin=0, end = 30, parameters=None)
# traci.vehicle.subscribe(vehicles[0], (traci.VAR_FOLLOWER,), 0, 30, {traci.VAR_FOLLOWER: ("d", 10)})

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
        if vehicle not in addedVehciles:
            add_aggressive_behavior(vehicle)
            addedVehciles.append(vehicle)
    return addedVehciles

def contextSubscription(vehicle):
    desiredRange = 20
    traci.vehicle.subscribeContext(vehicle, traci.constants.CMD_GET_VEHICLE_VARIABLE, desiredRange)

def getSubscriptionResults(vehicle):
    res = traci.vehicle.getContextSubscriptionResults(vehicle)
    print(res)


addedVehciles = []
# edges = traci.ge
while step < 40:
    traci.simulationStep()
    time.sleep(1)
    addedVehciles = addAggressiveToAllVehicles(addedVehciles)
    if step == 8:
        traci.vehicle.setSpeed("flow1.0", 0)
        traci.vehicle.setColor("flow1.0", (255,0,0))

        traci.vehicle.setSpeed("flow2.0", 0)
        traci.vehicle.setColor("flow2.0", (255,0,0))
        # traci.vehicle.setRoute("flow2.2", traci.route.getEdges("r_1"))
    if step > 5:    
        traci.vehicle.changeLane("flow1.0", 1, 30)
        contextSubscription("flow2.2")
        getSubscriptionResults("flow2.2")
    collisions = traci.simulation.getCollisions()
    for collision in collisions:
        traci.vehicle.setSpeed(collision.collider, 0)
        traci.vehicle.setEmergencyDecel(collision.collider, 1000)
        traci.vehicle.setColor(collision.collider, (255,0,0))
        traci.vehicle.setSpeed(collision.victim, 0)
    
        
    step += 1
# collisions = traci.simulation.getCollisions()
traci.close()

