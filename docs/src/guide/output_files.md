# **Output files**
In QuanEstimation, the output data will be saved into files during or after the optimization process. There are currently two types of them: the values of the objective function and the values of variables from corresponding scheme (optimizations for controls, states, or measurements).  

For the objective fucntion, the optimized variables at each optimization step will be used to evaluate the objective function, and then the obtained values will be saved sequentially, each line a value, into a CSV file named `f.csv`. However, for the optimized variables, depending on the choice of algorithm and optimization scenario, the storage format of output data varies accordingly. This guide is intended to help users to understand how the optimized data is saved and how to extract and manipulate them for further research. 

## Control optimization
The control optimization results will be saved into a `.npy` file named `controls.npy`. If the argument `savefile` is set to `False`, the control coefficients after the final round of optimization will be saved, with the first dimension corresponding to the number of control, and the second dimension corresponding to the number of time intervals. Otherwise, all the control coefficients after each round of optimization will be saved. The saved control optimization results can be load with 
``` py
# load the controls
controls = np.load("controls.npy")
```

## State optimization
The state optimization results will be saved into a `.npy` file named `states.npy`. If the argument `savefile` is set to `False`, the final optimized state vectors will be saved. Otherwise, all the state vectors after each round of optimization will be saved. The saved state optimization results can be load with 
``` py
# load the states
states = np.load("states.npy")
```

## Measurement optimization
The measurement optimization results will be saved into a `.npy` file named `measurements.npy`. If the argument `savefile` is set to `False`, the final optimized POVMs will be saved. Otherwise, all the lists of POVMs after each round of optimization will be saved. The saved measurement optimization results can be load with 
``` py
# load the measurements
M = np.load("measurements.npy")
```