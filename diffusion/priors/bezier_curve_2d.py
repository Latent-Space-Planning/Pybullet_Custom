import numpy as np
import einops
import torch

def generate_trajectories(num_samples, traj_len, bounds = np.array([[-1, -1], [1, 1]])):
    
    start = np.random.uniform(low = bounds[0], high = bounds[1], size = (num_samples, 2))
    goal = np.random.uniform(low = bounds[0], high = bounds[1], size = (num_samples, 2))

    curve_points = generate_bezier_curves(num_samples, traj_len)

    line = goal - start

    dist = np.linalg.norm(line, axis = 1)
    scaled_curves = dist.reshape((-1, 1, 1)) * curve_points

    theta_list = -np.arctan2(line[:, 1], line[:, 0]).reshape((-1, 1))
    trajectories = np.zeros((num_samples, 2, traj_len))

    # scaled_curves => num_samples x 2 x traj_len
    # theta_list    => num_samples
    # trajectories  => num_samples x 2 x traj_len

    trajectories[:, 0, :] = scaled_curves[:, 0, :] * np.cos(theta_list) + scaled_curves[:, 1, :] * np.sin(theta_list) + start[:, 0].reshape((-1, 1))
    trajectories[:, 1, :] = -1 * scaled_curves[:, 0, :] * np.sin(theta_list) + scaled_curves[:, 1, :] * np.cos(theta_list) + start[:, 1].reshape((-1, 1))

    return torch.tensor(trajectories)

def generate_bezier_curves(num_samples, traj_len):

    # Define control points
    P0 = np.zeros((num_samples, 2))
    P3 = np.repeat([[1, 0]], num_samples, axis = 0)

    P1 = np.random.uniform(low = [0, -1], high = [1, 1], size = (num_samples, 2))

    P2_low = np.concatenate([[P1[:, 0]], np.zeros((1, num_samples))]).T
    P2_high = np.concatenate([np.ones((1, num_samples)), np.sign([P1[:, 1]])]).T

    P2 = np.random.uniform(low = P2_low, high = P2_high, size = (num_samples, 2))

    # Define t values (parameter values)
    t = np.linspace(0, 1, traj_len)

    # Define the bezier curve formula
    B = lambda t: (1-t)**3*P0 + 3*t*(1-t)**2*P1 + 3*t**2*(1-t)*P2 + t**3*P3

    # # Generate the bezier curve points
    curve_points = np.array([B(t_i) for t_i in t])
    curve_points = einops.rearrange(curve_points, 'n b c -> b c n')

    return curve_points.copy()