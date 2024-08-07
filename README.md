# Thermal-PINO
This is a fully executable training code for an augmented Fourier Neural Operator for 2D conjugate heat transfer problems. See channel.py to train a model. If you already have velocity data, you can use heatgen.py to generate temperature data from the heat convection-diffusion equation.
If not, please use some of the OpenFOAM functionality to generate data.

![image](https://github.com/rishiiyer01/Thermal-PINO/assets/79063239/f1cee470-3fbd-455c-b9a3-a0cfa21a1361)
Full model was trained on a single NVIDIA p100 15 gb node. Inference takes less than 0.1s, this type of architecture shows potential to automate ChT simulations of varying similar geometries.


See inference_testing_notebook.ipynb for test-set examples.




Paper:  https://rishiiyer.com/THESIS_RISHABH_IYER.pdf
