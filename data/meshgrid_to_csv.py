##convert numpy to openfoam 

import numpy as np

# Load your meshgrid data
X_data = np.load("Pipe_X.npy")
Y_data = np.load("Pipe_Y.npy")

# Select the desired meshgrid
randPipeX = X_data[45]
randPipeY = Y_data[45]

# Stack the X and Y coordinates
points = np.stack((randPipeX.flatten(), randPipeY.flatten()), axis=-1)

# Save the points to a CSV file
np.savetxt("meshgrid.csv", points, delimiter=",", fmt="%.6f")

#I moved the csv to the relevant openfoam dir, you have to download data from google drive