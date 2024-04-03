# Thermal-PINO
Here, we are using physics informed neural operator to accelerate simulations for conjugate heat transfer problems. This requires passing a signed distance field or a mask(s) as input to the neural operator to guide the model to understand which parts of the domain are fluid vs solid. For now, we are completing the work for a simple irregular 2d channel of water inside of a 2d block of aluminum surrounded by free convection room temperature air. Data generation is partially from OpenFOAM's chtsolver, and partially from a JAX numerical solver I programmed for situations where the velocity field at steady state is already known. 


fully executable training and inference code is coming soon.
