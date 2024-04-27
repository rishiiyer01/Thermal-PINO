# Thermal-PINO
This is a fully executable training code for an augmented Fourier Neural Operator for 2D conjugate heat transfer problems. See channel.py to train a model. If you already have velocity data, you can use heatgen.py to generate temperature data from the heat convection-diffusion equation.
If not, please use some of the OpenFOAM functionality to generate data.

Full model was trained on a single NVIDIA p100 15 gb node. 
