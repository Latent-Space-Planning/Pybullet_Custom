from pybullet_environment import *
from diffusion_model import *

import os
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- User Input ------------------- #

traj_len = 50
T = 255
num_channels = 7

start = np.array([0.5,  0.3,  0.2])
start_orientation = np.array([3.14, 0, 1.57])

goal = np.array([0.5,  0.3,  0.2])
goal_orientation = np.array([3.14, 0, 1.57])

model_name = "./diffusion_model/model_weights/7dof/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)
# --------------------------------------------------- #

if not os.path.exists(model_name):
    print("Model does not exist for these parameters. Train a model first.")
    _ = input("Press anything to exit")
    exit()

env = RobotEnvironment()

diffuser = Diffusion(T)

denoiser = TemporalUNet(model_name = model_name, input_dim = num_channels, time_dim = 32, dims=(32, 64, 128, 256, 512, 512))
_ = denoiser.to(device)

# start_joint = env.inverse_kinematics(start, start_orientation)
goal_joint = env.inverse_kinematics(goal, goal_orientation)

traj = np.load("traj.npy")
start_joint = np.append(traj[0, :], [0., 0.]) #np.array([0.7, -0.2, 0.3, -1.8, 0., 2.1, 1.57, 0., 0.])
goal_joint = np.append(traj[-1, :], [0., 0.]) #np.array([-0.2, -0.2, -0.7, -1.8, 0., 2.1, 1.57, 0., 0.])

os.system("clear")
print("Environment and Model Loaded \n")

_ = input("Press Enter to start generating trajectory")

st_time = time.time()

trajectory = diffuser.denoise(model = denoiser,
                              traj_len = traj_len,
                              num_channels = num_channels,
                              condition = True,
                              start = start_joint,
                              goal = goal_joint)

print("Denoising took " + str(np.round((time.time() - st_time), 2)) + " seconds")

env.execute_trajectory(trajectory)

_ = input("Execution complete. Press Enter to Continue")








