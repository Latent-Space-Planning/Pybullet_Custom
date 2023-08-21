from pybullet_environment import *
from diffusion_model import *
from guide import *

import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- User Input ------------------- #

traj_len = 50
T = 255
num_channels = 7

mpinet_index = 22498
# 7584 is failing
data = 'val'

obstacle_clearance = np.linspace(0.01, 0.15, 255)

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

obstacle_config, cuboid_config, cylinder_config, start_joints, goal_joints = env.get_mpinet_scene(mpinet_index, data)

guide = IntersectionVolumeGuide(env, obstacle_config, device, clearance = obstacle_clearance)

env.spawn_cuboids(cuboid_config)
env.spawn_cylinders(cylinder_config)

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

env.execute_trajectory(trajectory[0])

_ = input("Execution complete. Press Enter to Continue")
np.save(trajectory, "traj.npy")








