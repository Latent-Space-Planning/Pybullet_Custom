import numpy as np
import einops
import torch

def generate_trajectories(num_samples, traj_len, bounds = np.array([[-1, -1], [1, 1]])):
    
    start = np.random.uniform(low = bounds[0], high = bounds[1], size = (num_samples, 2))
    goal = np.random.uniform(low = bounds[0], high = bounds[1], size = (num_samples, 2))

    trajectories = np.zeros((num_samples, traj_len, 2))

    step_lim = np.array([0.001, 0.005])
    angle_lim = np.array([-np.pi/10, np.pi/10])

    trajectories[:, 0, :] = np.random.uniform(bounds[0], bounds[1], size = (num_samples, 2))
    direction = np.random.uniform(-np.pi, np.pi, size = (num_samples, ))
    step = np.random.uniform(step_lim[0], step_lim[1], size = (num_samples, ))
    delta = np.array([step * np.cos(direction), step * np.sin(direction)]).T
    trajectories[:, 1, :] = trajectories[:, 0, :] + delta
    prev_dir = direction.copy()

    for i in range(2, traj_len):
    
        direction = np.random.uniform(angle_lim[0], angle_lim[1], size = (num_samples, )) + prev_dir
        step = np.random.uniform(step_lim[0], step_lim[1], size = (num_samples, ))
        delta = np.array([step * np.cos(direction), step * np.sin(direction)]).T
        
        trajectories[:, i, :] = trajectories[:, i-1, :] + delta
    
        prev_dir = direction.copy()

    # trajectories = torch.tensor(trajectories)
    trajectories = einops.rearrange(trajectories, 'b n c -> b c n')

    return trajectories