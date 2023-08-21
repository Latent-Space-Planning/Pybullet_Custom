from pybullet_environment import *
from diffusion_model import *
from guide import *

import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- User Input ------------------- #

traj_len = 50
T = 255
num_channels = 7

scene_name = "two_blocks"
example_number = 2

obstacle_clearance = 0.15

model_name = "./diffusion_model/model_weights/7dof/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)
# --------------------------------------------------- #

if not os.path.exists(model_name):
    print("Model does not exist for these parameters. Train a model first.")
    _ = input("Press anything to exit")
    exit()

start_goal = np.load("scenes/" + scene_name + "/start_goals/example" + str(example_number) + ".npy")
start_joints = start_goal[0]
goal_joints = start_goal[1]

# Load Models:
env = RobotEnvironment()
diffuser = Diffusion(T, device = device)
denoiser = TemporalUNet(model_name = model_name, input_dim = num_channels, time_dim = 32, dims=(32, 64, 128, 256, 512, 512),
                        device = device)

obstacle_config = np.load("scenes/" + scene_name + "/obstacle_config.npy")
guide = IntersectionVolumeGuide(env, obstacle_config, device, clearance = obstacle_clearance)

env.spawn_cuboids(obstacle_config)

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








