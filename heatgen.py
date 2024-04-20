import numpy as np
import torch
import jax.numpy as jnp
from jax import jit, grad, vmap
from scipy.spatial import KDTree
from jax.experimental import sparse
import jax
from functools import partial
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import multiprocessing
import time


#class for generating heat eq solution based on aluminum and liquid water inlet, bcs are not configurable at this stage
class Heat_eq_generation():
    def __init__(self, x_channel, y_channel, velocity_data, domain_size, grid_res):
        start_time=time.time()
        self.x_channel = x_channel
        self.y_channel = y_channel
        coords_list = []
        for col in np.arange(x_channel.shape[1]):
            coords_list = coords_list + list(zip(x_channel[col, :], y_channel[col, :]))
        coords = jnp.array(coords_list)
        self.tree = KDTree(coords)
        Lx, Ly = domain_size
        Nx, Ny = grid_res
        nx, ny= grid_res
        x = jnp.linspace(0, Lx, Nx)
        y = jnp.linspace(-Ly, Ly, Ny)
        self.dx = Lx / Nx
        self.dy = 2 * Ly / Ny
        self.xgrid, self.ygrid = jnp.meshgrid(x, y)
        end_time=time.time()
        print(f"Initialization time: {end_time - start_time} seconds")
        
        start_time=time.time()
        self.u,self.p = self.interpolate(jax.lax.stop_gradient(velocity_data), jax.lax.stop_gradient(self.xgrid), jax.lax.stop_gradient(self.ygrid))
        self.u=self.u*4
        self.mask = self.map_channel_geometry(self.xgrid, self.ygrid)
        alpha_aluminum = 64 * 10**-6
        alpha_water = 0.143 * 10**-6
        alpha_field = jnp.where(self.mask == True, alpha_water, alpha_aluminum)
        end_time=time.time()
        print(f"Interpolation time: {end_time - start_time} seconds")

        
        self.assemble_coefficients_matrix_vmapped = self.assemble_coefficients_matrix
        self.apply_boundary_conditions_vmapped = self.apply_boundary_conditions
        self.T = self.solve_steady_state(jax.lax.stop_gradient(alpha_field), jax.lax.stop_gradient(self.u[:,:,0]), jax.lax.stop_gradient(self.u[:,:,1]), jax.lax.stop_gradient(self.mask), nx, ny, self.dx, self.dy)
        

    def is_inside_channel(self, x, y):
        inside = False
        nearest_dist, nearest_idx = self.tree.query((x, y), 1)
        nearest_point = self.tree.data[nearest_idx]
        nearest_x, nearest_y = nearest_point

        # Find the indices where the nearest x-coordinate occurs in the meshgrid
        x_indices = np.where(self.x_channel == nearest_x)

        if len(x_indices[0]) > 0:
            # Check if the y-coordinate is within the range of y-coordinates at the nearest x-coordinate
            row_index, col_index = x_indices[0][0], x_indices[1][0]
            if self.y_channel[row_index, col_index] <= y <= self.y_channel[row_index, -1]:
                inside = True

        return inside

    def map_channel_geometry(self, X, Y):
        # Create a mask array to represent the channel geometry
        mask = np.zeros_like(X, dtype=bool)

        # Iterate over the grid points and determine if each point lies inside the channel
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                mask[i, j] = self.is_inside_channel(X[i, j], Y[i, j])

        return mask

    def interpolate(self, velocity_data, xgrid, ygrid):
        grid_points = jnp.stack((xgrid, ygrid), axis=-1)

        # Perform KDTree queries for all grid points
        distances, indices = self.tree.query(grid_points, k=4)

        # Parallelize the interpolation using vmap and jit
        interpolate_vectorized = jit(vmap(vmap(lambda point, dist, idx: self.interpolate_point(point, dist, idx, velocity_data), in_axes=(0, 0, 0)), in_axes=(0, 0, 0)))
        data = interpolate_vectorized(grid_points, distances, indices)
        u=data[:,:,:2]
        p=data[:,:,2]
        
        return u,p

    @staticmethod
    @jit
    def interpolate_point(point, distances, indices, velocity_data):
        weights = 1.0 / distances
        weights /= jnp.sum(weights)
    
        velocity_sum = jnp.zeros(3)
        for k in range(4):
            idx = indices[k]
            velocity_sum = velocity_sum.at[0].add(weights[k] * jnp.take(velocity_data[0], idx))
            velocity_sum = velocity_sum.at[1].add(weights[k] * jnp.take(velocity_data[1], idx))
            velocity_sum=  velocity_sum.at[2].add(weights[k] * jnp.take(velocity_data[2], idx))
            
        return velocity_sum
        
    
    
    @staticmethod
    def assemble_coefficients_matrix(A, alpha_field, velocity_field_x, velocity_field_y, nx, ny, dx, dy, idx_range):
        alpha_field = jax.lax.stop_gradient(alpha_field)
        velocity_field_x = jax.lax.stop_gradient(velocity_field_x)
        velocity_field_y = jax.lax.stop_gradient(velocity_field_y)
        
        def body_fun(carry, idx):
            A = carry
            i, j = jnp.divmod(idx, ny)
            A = jax.lax.stop_gradient(A)
            
            def true_fn(A):
                index = j * nx + i
                
                # Diffusion terms
                A = A.at[index, index].add(-2 * alpha_field[i, j] / (dx**2) - 2 * alpha_field[i, j] / (dy**2))
                A = A.at[index, index - 1].add(alpha_field[i-1, j] / (dy**2))
                A = A.at[index, index + 1].add(alpha_field[i+1, j] / (dy**2))
                A = A.at[index, index - ny].add(alpha_field[i, j-1] / (dx**2))
                A = A.at[index, index +ny].add(alpha_field[i, j+1] / (dx**2))
        
                # Advection terms
                A = jax.lax.cond(
                    velocity_field_x[i, j] > 0,
                    lambda A: A.at[index, index].add(-velocity_field_x[i, j] / dx).at[index, index - ny].add(velocity_field_x[i, j] / dx),
                    lambda A: A.at[index, index].add(velocity_field_x[i, j] / dx).at[index, index + ny].add(-velocity_field_x[i, j] / dx),
                    A
                )
                
                A = jax.lax.cond(
                    velocity_field_y[i, j] > 0,
                    lambda A: A.at[index, index].add(-velocity_field_y[i, j] / dy).at[index, index - 1].add(velocity_field_y[i, j] / dy),
                    lambda A: A.at[index, index].add(velocity_field_y[i, j] / dy).at[index, index + 1].add(-velocity_field_y[i, j] / dy),
                    A
                )
                
                return A
            
            def false_fn(A):
                return A
            
            A = jax.lax.cond(
                jnp.logical_and(i >= 1, jnp.logical_and(i < nx - 1, jnp.logical_and(j >= 1, j < ny - 1))),
                true_fn,
                false_fn,
                A
            )
            
            return A, None
    
        A, _ = jax.lax.scan(body_fun, A, idx_range)
        
        return A
        
    
    @staticmethod
    def apply_boundary_conditions(A, b, mask, nx, ny, dx, dy):
        # Bottom boundary (i == 0)
        i = 0
        for j in range(ny):
            index = j * nx + i
            A = A.at[index, index].set(20 / 237 + 1 / dy)
            A = A.at[index, index + 1].set(20 / 237 - 1 / dy)
            b = b.at[index].set(400 / 237)
    
        # Top boundary (i == nx - 1)
        i = nx - 1
        for j in range(ny):
            index = j * nx + i
            A = A.at[index, index].set(20 / 237 - 1 / dy)
            A = A.at[index, index - 1].set(20 / 237 + 1 / dy)
            b = b.at[index].set(400 / 237)
    
        # Left boundary (j == 0)
        j = 0
        for i in range(nx):
            index = j * nx + i
            if mask[i, j]:
                A = A.at[index, index].set(1)
                b = b.at[index].set(1000)
            else:
                A = A.at[index, index].set(20 / 237 + 1 / dx)
                A = A.at[index, index + ny].set(20 / 237 - 1 / dx)
                b = b.at[index].set(400 / 237)
    
        # Right boundary (j == ny - 1)
        j = ny - 1
        for i in range(nx):
            index = j * nx + i
            if mask[i, j]:
                A = A.at[index, index].set(1 / dx)
                A = A.at[index, index - ny].set(-1 / dx)
                b = b.at[index].set(0)
            else:
                A = A.at[index, index].set(20 / 237 + 1 / dx)
                A = A.at[index, index - ny].set(20 / 237 - 1 / dx)
                b = b.at[index].set(400 / 237)
    
        return A, b
    
    def solve_steady_state(self, alpha_field, velocity_field_x, velocity_field_y, mask, nx, ny, dx, dy):
        start_time = time.time()
        A = jnp.zeros((nx * ny, nx * ny))
        A=jax.lax.stop_gradient(A)
        idx_range = jnp.arange(nx * ny)  # Create the range array outside the JIT-compiled function
        A = self.assemble_coefficients_matrix_vmapped(A, alpha_field, velocity_field_x, velocity_field_y, nx, ny, dx, dy, idx_range)
        b = jax.lax.stop_gradient(jnp.zeros(nx * ny))
        end_time = time.time()
        print(f"Matrix assembly time: {end_time - start_time} seconds")
    
        start_time = time.time()
        A, b = self.apply_boundary_conditions_vmapped(A, b, mask, nx, ny, dx, dy)
        A_sp = jax.experimental.sparse.csr_fromdense(A)
        end_time = time.time()
        print(f"bc+sparsify time: {end_time - start_time} seconds")
    
        start_time = time.time()
        
        T = jax.experimental.sparse.linalg.spsolve(A_sp.data,A_sp.indices,A_sp.indptr, b).reshape((ny, nx)).T
        end_time = time.time()
        print(f"solution time: {end_time - start_time} seconds")
        return T



import torch

X_data = np.load("/home/iyer.ris/Pipe_X.npy")
Y_data = np.load("/home/iyer.ris/Pipe_Y.npy")
velocity_data=np.load("/home/iyer.ris/pipe/Pipe_Q.npy")
randPipeX = X_data[45]
randPipeY = Y_data[45]
randQ=velocity_data[45]
print(randQ.shape)
#print(randPipeX[20,0],randPipeY[20,0],randQ[2,-1,:])
print(jax.devices())

input_x=torch.tensor(X_data)
input_y=torch.tensor(Y_data)
print(input_x.shape,input_y.shape)
input_x.unsqueeze(1)
input_y.unsqueeze(1)
INPUT=torch.stack((input_x,input_y),dim=1)
print(INPUT.shape)
velocities=torch.tensor(velocity_data)
print(velocities.shape)
#data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(INPUT[:1000], velocities[:1000]), batch_size=1, shuffle=False)


# Check if storage files exist, otherwise create new ones
if os.path.exists('u_store.npy'):
    u_store = torch.from_numpy(np.load('u_store.npy'))
else:
    u_store = torch.zeros_like(torch.stack((input_x, input_y), dim=1))

if os.path.exists('xgrid_store.npy'):
    xgrid_store = torch.from_numpy(np.load('xgrid_store.npy'))
else:
    xgrid_store = torch.zeros((1000, 1, 129, 129))

if os.path.exists('ygrid_store.npy'):
    ygrid_store = torch.from_numpy(np.load('ygrid_store.npy'))
else:
    ygrid_store = torch.zeros((1000, 1, 129, 129))

if os.path.exists('mask_store.npy'):
    mask_store = torch.from_numpy(np.load('mask_store.npy'))
else:
    mask_store = torch.zeros((1000, 1, 129, 129))

if os.path.exists('p_store.npy'):
    p_store = torch.from_numpy(np.load('p_store.npy'))
else:
    p_store = torch.zeros((1000, 1, 129, 129))

if os.path.exists('T_store.npy'):
    T_store = torch.from_numpy(np.load('T_store.npy'))
else:
    T_store = torch.zeros((1000, 1, 129, 129))

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True, help='Start index of the data subset')
parser.add_argument('--end', type=int, required=True, help='End index of the data subset')
args = parser.parse_args()

start_index = args.start
end_index = args.end

for i in range(start_index, end_index):
    x_channel = input_x[i].numpy()
    y_channel = input_y[i].numpy()
    velocity_data = velocities[i].numpy()

    domain_size = (10, 2.0)  # Example domain size
    grid_res = (100, 100)  # Example grid resolution

    obj = Heat_eq_generation(x_channel, y_channel, velocity_data, domain_size, grid_res)
    T = obj.solve_steady_state(obj.alpha_field, obj.u[:,:,0], obj.u[:,:,1], obj.mask, *grid_res, obj.dx, obj.dy)

    xgrid_store[i] = obj.xgrid
    ygrid_store[i] = obj.ygrid
    T_store[i] = T
    u_store[i] = obj.u
    mask_store[i] = obj.mask
    p_store[i] = obj.p

# Save the updated storage files
np.save('xgrid_store.npy', xgrid_store.numpy())
np.save('ygrid_store.npy', ygrid_store.numpy())
np.save('T_store.npy', T_store.numpy())
np.save('u_store.npy', u_store.numpy())
np.save('mask_store.npy', mask_store.numpy())
np.save('p_store.npy', p_store.numpy())


