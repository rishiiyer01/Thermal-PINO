# Thermal-PINO
Here, we are using physics informed neural operator to accelerate simulations in the complete absence of data. We are trying to generalize across parameters and geometries of the conjugate heat transfer problem using a variety of methods:
1. Lattice Boltzmann and SIMPLE-NS PINO
   a. As of right now, I support the iterative SIMPLE algorithm for channel flows of varying geometries. It is fully differentiable, so it is considered a physics informed loss with the benefits of data based losses.
3. Utilizing iterative based algorithms that use the neural operator ansatz to initialize the distribution, then computing the loss based on the difference between the initialization and the converged solution, which is faster than generating the data separately
A fully executable training and inference code is coming soon.
