import numpy as np
import einops
import torch

def generate_trajectories(num_samples, traj_len, bounds = np.array([[-1, -1], [1, 1]])):
    
    start = np.random.uniform(low = bounds[0], high = bounds[1], size = (num_samples, 2))
    goal = np.random.uniform(low = bounds[0], high = bounds[1], size = (num_samples, 2))

    # start = np.repeat(np.array([[-0.1, -0.3]]), num_samples, axis = 0)
    # goal = np.repeat(np.array([[0.6, 0.]]), num_samples, axis = 0)

    n = traj_len

    line = goal - start

    dist = np.linalg.norm(line, axis = 1)

    point_dists = np.linspace(0., dist, n).T

    theta = np.arctan2(line[:, 1], line[:, 0])
    theta = np.expand_dims(theta, axis = 1)

    delta_x = np.multiply(point_dists, np.cos(theta))
    delta_y = np.multiply(point_dists, np.sin(theta))

    delta_x = np.expand_dims(delta_x, axis = 2)
    delta_y = np.expand_dims(delta_y, axis = 2)

    delta = np.concatenate([delta_x, delta_y], axis = 2)

    trajectories = np.expand_dims(start, axis = 1) + delta

    trajectories = torch.tensor(trajectories)
    trajectories = einops.rearrange(trajectories, 'b n c -> b c n')

    return trajectories