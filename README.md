# Thermal-PINO
Here, we are using physics informed neural operator to accelerate simulations in the complete absence of data. We are trying to generalize across parameters and geometries of the conjugate heat transfer problem using a variety of methods:
1. Lattice Boltzmann and SIMPLE-NS PINO
2. Utilizing iterative based algorithms that use the neural operator ansatz to initialize the distribution, then computing the loss based on the difference between the initialization and the converged solution, which is faster than generating the data separately