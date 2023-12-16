# Enhancing Road Safety using V2V Simulations

This project focuses mainly on how to utilize the information shared from the nearby vehicles to actively avoid collisions, traffic jams by Changing speed, lanes and routes. We used Reinforcement Learning to train the model.

## Environment
To train the model, we have used the SUMO(Simulation of Urban MObility) environment.

## Files
1. run.py
    - This file acts as the controller. It controls the number of episodes for which the model has to train. The batch size of the replay memory is defined here.

2. Simulation.py
    - This file controls the simulation. It loads the vehicles, gets the states of the vehicles and nearby vehicles, gives the state to the model to get the action, performs the action and calculates the reward. At the end of the simulation, the model will get retrained based on the information stored in the memory.

3. DQNModeltorch.py
    - This file defines the DQN model. The Deep Q network and the learning rate is defined in this file. The rewards and penalties of the models are also defined in this file.

4. demo2.net.xml
    - This file defines the edges (roads), junctions and their connections.

5. demo2.rou.xml
    - This file defines the routes and vehicles. The number of vehicles and their arrival lane is defined here.     

6. demo2.sumocfg
    - The sumo configuration file loads the visualization of the network. To view the netowrk, open SUMO and load this file. Set the delay to 100 and click run to view the vehicles(defined in the demo2.rou.xml) go.

## Execution
1. Install the requirements stated in the requirements.txt or type the command in the terminal 
```pip install -r requirements.txt```
2. Follow this documentation to install the SUMO based on the OS: https://sumo.dlr.de/docs/Installing/index.html
3. Execute run.py or type the command in the terminal 
```python3 run.py```


## Demo
View the demos in the demos folder.

For more details, please refer project report.