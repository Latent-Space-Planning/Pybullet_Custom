from pybullet_environment import *
from diffusion_model import *
from guide import *

import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- User Input ------------------- #

traj_len = 50
T = 255
num_channels = 7

# start_joints = np.array([-0.571, 0.522, -0.134, -1.4, 0.341, 1.7, 0.572])
# goal_joints = np.array([0.9798614, 0.5573511, -0.1629493, -1.9912002, -0.23168434, 2.214461, -0.38091224])

start_joints = np.array([-0.371, 0.522, -0.134, -1.4, 0.341, 1.7, 0.572])
goal_joints = np.array([0.55, 0.5573511, -0.1629493, -1.9912002, -0.23168434, 2.214461, -0.38091224])

obstacle_clearance = 0.04

model_name = "./diffusion_model/model_weights/7dof/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)
# --------------------------------------------------- #

if not os.path.exists(model_name):
    print("Model does not exist for these parameters. Train a model first.")
    _ = input("Press anything to exit")
    exit()

# Load Models:
env = RobotEnvironment()
diffuser = Diffusion(T, device = device)
denoiser = TemporalUNet(model_name = model_name, input_dim = num_channels, time_dim = 32, dims=(32, 64, 128, 256, 512, 512),
                        device = device)

# Define Scene:
obstacle_config = np.zeros((1, 10))

# Obstacle A:
obs_pos = np.array([0.6, 0., 0.18])
obs_quat = np.array(env.client_id.getQuaternionFromEuler(np.array([0., 0., 0.])))
obs_size = np.array([0.5, 0.4, 0.36])
obstacle_config[0] = np.array([*obs_pos, *obs_quat, *obs_size])

guide = IntersectionVolumeGuide(env, obstacle_config, device, clearance = obstacle_clearance)

env.spawn_obstacles(obstacle_config)

# Define Environment:
# obs_pos = np.array([0.6, 0., 0.18])
# obs_ori = np.array([0., 0., 0.])
# obs_quat = np.array(env.client_id.getQuaternionFromEuler(obs_ori))
# obs_pose = np.array([*obs_pos, *obs_quat])

# obs_size = np.array([0.3, 0.4, 0.36])

# obs_pos = np.array([0.6, 0., 0.24])
# obs_ori = np.array([0., 0., 0.])
# obs_quat = np.array(env.client_id.getQuaternionFromEuler(obs_ori))
# obs_pose = np.array([*obs_pos, *obs_quat])

# obs_size = np.array([0.5, 0.06, 0.48])

# obs_pos = np.array([0.7, 0., 0.6])
# obs_ori = np.array([0., 0., 0.])
# obs_quat = np.array(env.client_id.getQuaternionFromEuler(obs_ori))
# obs_pose = np.array([*obs_pos, *obs_quat])

# obs_size = np.array([0.3, 0.06, 1.2])

os.system("clear")
print("Environment and Model Loaded \n")

_ = input("Press Enter to start generating trajectory")

st_time = time.time()

trajectory = diffuser.denoise_guided(model = denoiser,
                                     guide = guide,
                                     traj_len = traj_len,
                                     num_channels = num_channels,
                                     condition = True,
                                     start = start_joints,
                                     goal = goal_joints)

print("Denoising took " + str(np.round((time.time() - st_time), 2)) + " seconds")

env.execute_trajectory(trajectory)

_ = input("Execution complete. Press Enter to Continue")
np.save(trajectory, "traj.npy")








