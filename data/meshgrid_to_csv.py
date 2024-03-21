import numpy as np
from stl import mesh

# Load your meshgrid data
X_data = np.load("Pipe_X.npy")
Y_data = np.load("Pipe_Y.npy")

# Select the desired meshgrid
randPipeX = X_data[45]
randPipeY = Y_data[45]

# Create a triangular mesh from the meshgrid
vertices = np.zeros((randPipeX.shape[0], randPipeX.shape[1], 3))
vertices[:, :, 0] = randPipeX
vertices[:, :, 1] = randPipeY

faces = np.zeros((2 * (randPipeX.shape[0] - 1) * (randPipeX.shape[1] - 1), 3), dtype=np.int64)
for i in range(randPipeX.shape[0] - 1):
    for j in range(randPipeX.shape[1] - 1):
        face_idx = 2 * (i * (randPipeX.shape[1] - 1) + j)
        faces[face_idx] = [i * randPipeX.shape[1] + j, (i + 1) * randPipeX.shape[1] + j, i * randPipeX.shape[1] + j + 1]
        faces[face_idx + 1] = [(i + 1) * randPipeX.shape[1] + j, (i + 1) * randPipeX.shape[1] + j + 1, i * randPipeX.shape[1] + j + 1]

# Create an STL mesh
stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i in range(faces.shape[0]):
    for j in range(3):
        stl_mesh.vectors[i, j] = vertices[faces[i, j] // randPipeX.shape[1], faces[i, j] % randPipeX.shape[1]]

# Save the STL file
stl_mesh.save("/home/iyer.ris/Thermal-PINO/conjugateHeatTransfer/system/meshgrid.stl")