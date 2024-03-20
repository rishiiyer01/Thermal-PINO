import numpy as np

# Load your meshgrid data
X_data = np.load("Pipe_X.npy")
Y_data = np.load("Pipe_Y.npy")

# Select the desired meshgrid
randPipeX = X_data[45]
randPipeY = Y_data[45]

# Calculate the cell size
cell_size_x = (randPipeX.max() - randPipeX.min()) / (randPipeX.shape[0] - 1)
cell_size_y = (randPipeY.max() - randPipeY.min()) / (randPipeY.shape[1] - 1)

# Create cell centers by adding half the cell size to the vertices
cell_centers_x = randPipeX + cell_size_x / 2
cell_centers_y = randPipeY + cell_size_y / 2

# Stack the X and Y coordinates of cell centers
cell_centers = np.stack((cell_centers_x.flatten(), cell_centers_y.flatten()), axis=-1)

# Save the points to a CSV file
np.savetxt("/home/iyer.ris/Thermal-PINO/conjugateHeatTransfer/system/meshgrid.csv", cell_centers, delimiter=",", fmt="%.6f")

