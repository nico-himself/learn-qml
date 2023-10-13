'''
Qubit rotation example from the PennyLane documentation.
src: https://pennylane.ai/qml/demos/tutorial_qubit_rotation

This example shows how to use PennyLane to optimize the rotation angles of a single qubit. This is
akin to the "Hello World" of quantum computing. The goal is to find the rotation angles that
minimize the expectation value of the PauliZ observable. This is a simple example of a variational
quantum algorithm, where the quantum circuit is used as a "black box" by a classical optimizer.
'''

import pennylane as qml
from pennylane import numpy as np

# create a device from which we can extract the circuit

'''
The device is a simulator or hardware that can run the circuit.
device("lightning.qubit", wires=1) creates a device with 1 wire and 1 qubit.
wires: sub-systems of the device.
'''
dev = qml.device("lightning.qubit", wires=1)

'''
Now that we've initialized our device, we can begin to 
construct a quantum node (or QNode), which is an abstract
encapsulation of a quantum function described by a quantum circuit.
'''
@qml.qnode(dev, interface="autograd")   # interface="autograd" is for automatic differentiation (AD)
                                        # interface="torch" is for PyTorch
                                        # @ is a decorator which is a function that takes another function as input
# The function below is the quantum circuit we turn into a QNode using the decorator
def circuit(params):
    qml.RX(params[0], wires=0) # RX is a rotation around the x-axis of the Bloch sphere
    qml.RY(params[1], wires=0) # RY is a rotation around the y-axis of the Bloch sphere
    return qml.expval(qml.PauliZ(0)) # expval is the expectation value of the observable PauliZ(0)

print(circuit([0.54, 0.12]))

dcircuit = qml.grad(circuit, argnum=0) # argnum=0 means we are taking the gradient with respect to the first argument
print(dcircuit([0.54, 0.12])) 

# same idea but with two arguments instead of one, and we want the gradient with respect to both arguments
@qml.qnode(dev, interface="autograd")
def circuit2(phi1, phi2):
    qml.RX(phi1, wires=0)
    qml.RY(phi2, wires=0)
    return qml.expval(qml.PauliZ(0))    # the result is the same as above because we are still only measuring
                                        # the expectation value of the PauliZ observable

dcircuit = qml.grad(circuit2, argnum=[0, 1])    # argnum=[0, 1] means we are taking the gradient with respect
                                                # to the first AND second arguments
print(dcircuit(0.54, 0.12))

# OPTIMIZATION
# now we want to optimize the parameters of the circuit to minimize the cost function
def cost(x):
    return circuit(x) # the cost function is the circuit itself (the expectation value of the PauliZ observable)

# choose some small initial parameters for phi1 and phi2
init_params = np.array([0.011, 0.012], requires_grad=True)
print(cost(init_params))

# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4) # stepsize is the learning rate (how big of a step we take in the gradient direction)

# set the number of steps
steps = 100
# set the initial parameter values
params = init_params

for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)

    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

print("Optimized rotation angles: {}".format(params))