import numpy as np
import math
import einops
import torch

def generate_trajectories(num_samples, traj_len, control_points = 10, bounds = np.array([[-1, -1], [1, 1]])):
    
    start = np.random.uniform(low = bounds[0], high = bounds[1], size = (num_samples, 2))
    goal = np.random.uniform(low = bounds[0], high = bounds[1], size = (num_samples, 2))

    curve_points = generate_bezier_curves(num_samples, traj_len, control_points)

    line = goal - start

    dist = np.linalg.norm(line, axis = 1)
    scaled_curves = dist.reshape((-1, 1, 1)) * curve_points / (curve_points[:, 0, -1].reshape((-1, 1, 1)))

    theta_list = -np.arctan2(line[:, 1], line[:, 0]).reshape((-1, 1))
    trajectories = np.zeros((num_samples, 2, traj_len))

    # scaled_curves => num_samples x 2 x traj_len
    # theta_list    => num_samples
    # trajectories  => num_samples x 2 x traj_len

    trajectories[:, 0, :] = scaled_curves[:, 0, :] * np.cos(theta_list) + scaled_curves[:, 1, :] * np.sin(theta_list) + start[:, 0].reshape((-1, 1))
    trajectories[:, 1, :] = -1 * scaled_curves[:, 0, :] * np.sin(theta_list) + scaled_curves[:, 1, :] * np.cos(theta_list) + start[:, 1].reshape((-1, 1))

    return torch.tensor(trajectories)

def bezier_function(t, P):

    out = 0
    n = P.shape[1]

    for i in range(n):

        out += P[:, i, :] * math.comb(n-1, i) * (t**i) * ((1-t)**(n-1-i))

    return out

def generate_bezier_curves(num_samples, traj_len, control_points = 5):
    
    # Control points array:
    P = np.zeros((num_samples, control_points, 2))

    # Define last points:
    P[:, -1, 0] = np.random.uniform(low = 0.002, high = 1, size = (num_samples,))
    P[:, -1, 1] = 0
    
    for i in range(1, control_points-1):
    
        P[:, i, 0] = np.random.uniform(low = P[:, i-1, 0], high = P[:, -1, 0])
        P[:, i, 1] = np.random.uniform(low = -1, high = 1, size = (num_samples,))

    # Define t values (parameter values)
    t = np.linspace(0, 1, traj_len)

    # # Generate the bezier curve points
    curve_points = np.array([bezier_function(t_i, P) for t_i in t])
    curve_points = einops.rearrange(curve_points, 'n b c -> b c n')

    return curve_points.copy()