from pybullet_environment import *
from diffusion_model import *
from guide import *

import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- User Input ------------------- #

traj_len = 50
T = 255
num_channels = 7

start_index = 7584         # First MPiNet trajectory index to process
num_runs = 3            # Till which trajectory to denoise
batch_size = 50         # Number of trajectories to denoise at once

run_number = 1

data = 'val'

obstacle_clearance = np.linspace(0.03, 0.15, 255)       # Obstacle clearance scheduled

model_name = "./diffusion_model/model_weights/7dof/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)
# --------------------------------------------------- #

if not os.path.exists(model_name):
    print("Model does not exist for these parameters. Train a model first.")
    _ = input("Press anything to exit")
    exit()

# Load Models:
env = RobotEnvironment(gui = False)
diffuser = Diffusion(T, device = device)
denoiser = TemporalUNet(model_name = model_name, input_dim = num_channels, time_dim = 32, dims=(32, 64, 128, 256, 512, 512),
                        device = device)

trajectory_results = np.zeros((num_runs, batch_size, num_channels, traj_len))
min_trajectories = np.zeros((num_runs, num_channels, traj_len))
metrics = np.zeros((num_runs, 7))

i = 0

for mpinet_index in range(start_index, start_index + num_runs):
    
    env.clear_obstacles()

    obstacle_config, cuboid_config, cylinder_config, start_joints, goal_joints = env.get_mpinet_scene(mpinet_index, data)

    guide = IntersectionVolumeGuide(env, obstacle_config, device, clearance = obstacle_clearance)

    env.spawn_collision_cuboids(cuboid_config)
    env.spawn_collision_cylinders(cylinder_config)

    # os.system("clear")
    # print("Environment and Model Loaded \n")

    # _ = input("Press Enter to start generating trajectory")

    st_time = time.time()

    trajectories = diffuser.denoise_guided(model = denoiser,
                                        guide = guide,
                                        batch_size = batch_size,
                                        traj_len = traj_len,
                                        num_channels = num_channels,
                                        condition = True,
                                        start = start_joints,
                                        goal = goal_joints)
    
    trajectory_results[i] = trajectories.copy()
    
    trajectory = guide.choose_best_trajectory(trajectories)
    # trajectory is (7, 50) numpy array
    min_trajectories[i] = trajectory.copy()

    end_time = time.time()
    
    success, joint_path_length, end_eff_path_length, joint_smoothness, end_eff_smoothness = env.benchmark_trajectory(trajectory, guide)
    
    metrics[i, 0] = mpinet_index
    metrics[i, 1] = success
    metrics[i, 2] = joint_path_length
    metrics[i, 3] = end_eff_path_length
    metrics[i, 4] = joint_smoothness
    metrics[i, 5] = end_eff_smoothness
    metrics[i, 6] = np.round((end_time - st_time), 2)

    np.save("results/trajectory_results" + str(run_number) + ".npy", trajectory_results)
    np.save("results/min_trajectories" + str(run_number) + ".npy", min_trajectories)
    np.save("results/metrics" + str(run_number) + ".npy", metrics)

    i += 1






