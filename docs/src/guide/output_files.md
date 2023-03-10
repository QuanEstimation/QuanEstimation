# **Output files**
In QuanEstimation, the output data will be saved into `.csv` files during or after the optimization process. There are currently two types of them: the values of the objective function and the values of variables from corresponding scheme (optimizations for controls, states, or measurements).  

For the objective fucntion, the optimized variables at each optimization step will be used to evaluate the objective function, and then the obtained values will be saved sequentially, each line a value, into a file named `f.csv`. However, for the optimized variables, depending on the choice of algorithm and optimization scenario, the storage format of output data varies accordingly. This guide is intended to help users to understand how the optimized data is saved and how to extract and manipulate them for further research. 

## Control optimization
``` py
# convert the ".csv" file to the ".npy" file
controls_ = np.loadtxt("controls.csv", dtype=np.complex128)
csv2npy_controls(controls_, num)
# load the controls
controls = np.load("controls.npy")
```

## State optimization
``` py
# convert the ".csv" file to the ".npy" file
states_ = np.loadtxt("states.csv", dtype=np.complex128)
csv2npy_states(states, num=1)
# load the states
states = np.load("states.npy")
```

## Measurement optimization
``` py
# convert the ".csv" file to the ".npy" file
M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
csv2npy_measurements(M_, M_num)
# load the measurements
M = np.load("measurements.npy")[-1]
```