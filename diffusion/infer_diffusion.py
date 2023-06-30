import torch
import time
import os
import matplotlib.pyplot as plt
import einops
import numpy as np
import pybullet as p

import sys 

sys.path.append('/home/vishal/Volume_E/Active/Undergrad_research/CoRL2023/codes/Vishal/Structured_Bullet/diffusion/Models')
sys.path.append('/home/vishal/Volume_E/Active/Undergrad_research/CoRL2023/codes/Vishal/Structured_Bullet/diffusion/priors')
sys.path.append('/home/vishal/Volume_E/Active/Undergrad_research/CoRL2023/codes/Vishal/Structured_Bullet/diffusion/scripts')

from diffusion.scripts.diffusion_functions import *
from diffusion.priors.bezier_curve_2d import *
from diffusion.Models.temporalunet import TemporalUNet

import h5py

def gen_trajectory(conditioning=False, start=np.zeros(7, ), goal=np.zeros(7, ), traj_count=0):
    '''
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------- Edit these parameters -------------- #
    traj_len = 50   
    T = 255
    batch_size = 2048

    # start =  [-0.5 ,  -0.1]
    # goal =  [0.3 , 0.3]
    # --------------------------------------------------- #

    #model_name = "Model_weights/straight_line/" + "TemporalUNetModel_Conditioned_T" + str(T) + "_N" + str(traj_len)
    model_name = "./diffusion/checkpoints/7dof/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)

    if not os.path.exists(model_name):
        print("Model does not exist for these parameters. Train a model first.")
        _ = input("Press anything to exit")
        exit()

    denoiser = TemporalUNet(model_name = model_name, input_dim = 7, time_dim = 32, dims=(32, 64, 128, 256, 512, 512))
    _ = denoiser.to(device)

    beta = schedule_variance(T)
    alpha = 1 - beta
    alpha_bar = np.array(list(map(lambda t:np.prod(alpha[:t]), np.arange(T+1)[1:])))

    mean = np.zeros(traj_len)
    cov = np.eye(traj_len)

    xT = np.random.multivariate_normal(mean, cov, (1, 7))
    X_t = xT.copy()

    # CONDITIONING:
    rest_poses = np.array([-0.00029003781167199756, -0.10325848749542864, 0.00020955647419784322,
                                    -3.0586072742527284, 1.3219639494062742e-05, 2.9560190438374425, 2.356101763339709])
    X_t[:, :, 0] = start[:] # start[:]
    X_t[:, :, -1] = goal[:]

    reverse_diff_traj = np.zeros((T+1, traj_len, 7))
    reverse_diff_traj[T] = einops.rearrange(xT[0], 'c n -> n c').copy()

    denoiser.train(False)

    for t in range(T, 0, -1):

        if t > 1:
            z = np.random.multivariate_normal(mean, cov, (1, 7))
        else:
            z = np.zeros((1, 7, traj_len))
        
        X_input = torch.tensor(X_t, dtype = torch.float32).to(device)
        time_in = torch.tensor([t], dtype = torch.float32).to(device)

        epsilon = denoiser(X_input, time_in).numpy(force=True)

        # posterier_mean = (1/np.sqrt(alpha[t-1])) * (X_t - ((1 - alpha[t-1])/(np.sqrt(1 - alpha_bar[t-1]))) * epsilon)
        # if t > 1:
        #     posterier_var = ((1 - alpha_bar[t-2]) / (1 - alpha_bar[t-1])) * beta[t-1]

        # for ch in range(X_t.shape[1]):
        #     X_t[0, ch, :] = np.random.multivariate_normal(mean = posterier_mean[0, ch, :], cov = beta[t-1] * cov)
        
        X_t = (1/np.sqrt(alpha[t-1])) * (X_t - ((1 - alpha[t-1])/(np.sqrt(1 - alpha_bar[t-1]))) * epsilon) + beta[t-1]*z
        
        # CONDITIONING:
        X_t[:, :, 0] = start[:] # rest_poses # start[:]
        X_t[:, :, -1] = goal[:]
        
        reverse_diff_traj[t-1] = einops.rearrange(X_t[0], 'c n -> n c').copy()

        # os.system("cls")
        print(f"\rDenoised to {t-1} steps", end="")

    np.save(f"/home/vishal/Volume_E/Active/Undergrad_research/CoRL2023/codes/Vishal/7DOF-Diffusion/results/7dof/generated_trajectories/reverse_trajectory_st_gl_fix_val_{traj_count}.npy", reverse_diff_traj)

def infer(denoiser, q0, target):
    '''Finds the trajectory using the denoiser between q0 and target

    Parameters:
    model: Diffusion model
    q0: Initial Joint Config
    target: Final Joint Config
    '''
    print("Hii")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------- Edit these parameters -------------- #
    traj_len = 50   
    T = 255 # 5

    beta = schedule_variance(T)
    alpha = 1 - beta
    alpha_bar = np.array(list(map(lambda t:np.prod(alpha[:t]), np.arange(T+1)[1:])))

    mean = np.zeros(traj_len)
    cov = np.eye(traj_len)

    xT = np.random.multivariate_normal(mean, cov, (1, 7))
    X_t = xT.copy()

    # CONDITIONING:
    rest_poses = np.array([-0.00029003781167199756, -0.10325848749542864, 0.00020955647419784322,
                                    -3.0586072742527284, 1.3219639494062742e-05, 2.9560190438374425, 2.356101763339709])
    X_t[:, :, 0] = q0 # start[:] # start[:]
    X_t[:, :, -1] = target # goal[:]

    reverse_diff_traj = np.zeros((T+1, traj_len, 7))
    reverse_diff_traj[T] = einops.rearrange(xT[0], 'c n -> n c').copy()

    denoiser.train(False)

    for t in range(T, 0, -1):
        print(f"\rTimestep: {t}", end="")

        if t > 1:
            z = np.random.multivariate_normal(mean, cov, (1, 7))
        else:
            z = np.zeros((1, 7, traj_len))
        
        X_input = torch.tensor(X_t, dtype = torch.float32).to(device)
        time_in = torch.tensor([t], dtype = torch.float32).to(device)

        epsilon = denoiser(X_input, time_in).numpy(force=True)
        
        X_t = (1/np.sqrt(alpha[t-1])) * (X_t - ((1 - alpha[t-1])/(np.sqrt(1 - alpha_bar[t-1]))) * epsilon) + beta[t-1]*z
        
        # CONDITIONING:
        X_t[:, :, 0] = q0 # start[:] # rest_poses # start[:]
        X_t[:, :, -1] = target # goal[:]
        
        reverse_diff_traj[t-1] = einops.rearrange(X_t[0], 'c n -> n c').copy()

    print("Done!!!")
    return reverse_diff_traj[0]

def clip_joints(q):
    lower_lims = np.array([-2.8973000049591064, -1.7627999782562256, -2.8973000049591064, -3.0717999935150146, -2.8973000049591064, -0.017500000074505806, -2.8973000049591064])
    upper_lims = np.array([2.8973000049591064, 1.7627999782562256, 2.8973000049591064, -0.0697999969124794, 2.8973000049591064, 3.752500057220459, 2.8973000049591064])
    return np.clip(q, lower_lims, upper_lims)

def value_func_and_grad(obstacle_center, link_pos):
    '''
    Returns:
    Value: Distance to the obstacle
    Grad: Gradient of Value w.r.t. pose of the link
    '''
    dist = np.linalg.norm(obstacle_center - link_pos) # **2
    if dist > 0.3:
        grad = np.zeros_like(link_pos)
    else:
        grad = 2*(link_pos - obstacle_center)
    return dist, grad.reshape((3, 1))

def calc_gradients(obstacle_center, traj, client_id, panda, panda_joints):
    '''Calculate Gradients
    '''
    gradients = np.zeros(traj.shape, dtype=float)
    for j in range(50):
        numJoints = p.getNumJoints(panda)
        for i in range(12):# numJoints):
            q = np.zeros(shape=(9, )) # + 0.02
            q[:7] = traj[0, :, j].reshape((7, )) # batch_num, joints, waypoint_num
            # print(q)
            # mpos, mvel, mtorq = getMotorJointStates(panda)
            # q = q0 = np.array([-0.5778415 ,  0.54352427,  0.6913696 , -2.5508945 , -0.11496189, 3.0910137, -2.49674   ])
            # numJoints = p.getNumJoints(panda)
            # ee_ind = numJoints - 1
            # print(ee_ind)
            # print(f'Panda id: {panda}')
            # print(np.array([0, 0, 0], dtype=float), q, np.zeros_like(q), np.zeros_like(q))
            clipped_q = np.zeros_like(q)
            clipped_q[:7] = clip_joints(q[:7])
            
            for i, joint_ind in enumerate(panda_joints):
                client_id.resetJointState(panda, joint_ind, clipped_q[i])

            link_pos = client_id.getLinkState(panda, i, computeForwardKinematics=1)[0]
            com_trn = client_id.getLinkState(panda, i, computeForwardKinematics=1)[2]
            J_t, J_r = client_id.calculateJacobian(panda, i, list(com_trn), clipped_q.tolist(), np.zeros_like(q).tolist(), np.zeros_like(q).tolist())
            J_t = np.array(J_t)

            value, grad = value_func_and_grad(obstacle_center, np.array(link_pos))# J_t@clipped_q.reshape((9, 1)))
            if np.linalg.norm(grad) > 0:
                print(f"Value: {value}, Grad: {grad}")
            # print("HIIIIIIIIIII")
            net_grad = np.sum(grad * J_t[:, :7], axis=0) # .reshape((traj[0, :, j].shape[0], traj[0, :, j].shape[1]))
            denom = np.linalg.norm(net_grad)
            if denom > 0:
                net_grad = net_grad / denom

            gradients[0, :, j] += net_grad.reshape((traj[0, :, j].shape))/7
    # print(f"Gradients: {gradients}")
    print(f"Norm of gradient: {np.linalg.norm(gradients)}")
    return np.clip(gradients, -1, 1)

def infer_guided(denoiser, q0, target, obstacle_center, client_id, panda, panda_joints):
    '''Finds the trajectory using the denoiser between q0 and target

    Parameters:
    model: Diffusion model
    q0: Initial Joint Config
    target: Final Joint Config
    '''
    print("Hii")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------- Edit these parameters -------------- #
    traj_len = 50   
    T = 255 # 5

    beta = schedule_variance(T)
    alpha = 1 - beta
    alpha_bar = np.array(list(map(lambda t:np.prod(alpha[:t]), np.arange(T+1)[1:])))

    mean = np.zeros(traj_len)
    cov = np.eye(traj_len)

    xT = np.random.multivariate_normal(mean, cov, (1, 7))
    X_t = xT.copy()

    # CONDITIONING:
    rest_poses = np.array([-0.00029003781167199756, -0.10325848749542864, 0.00020955647419784322,
                                    -3.0586072742527284, 1.3219639494062742e-05, 2.9560190438374425, 2.356101763339709])
    X_t[:, :, 0] = q0 # start[:] # start[:]
    X_t[:, :, -1] = target # goal[:]

    reverse_diff_traj = np.zeros((T+1, traj_len, 7))
    reverse_diff_traj[T] = einops.rearrange(xT[0], 'c n -> n c').copy()

    denoiser.train(False)

    for t in range(T, 0, -1):
        print(f"\rTimestep: {t}", end="")

        if t > 1:
            z = np.random.multivariate_normal(mean, cov, (1, 7))
        else:
            z = np.zeros((1, 7, traj_len))
        
        X_input = torch.tensor(X_t, dtype = torch.float32).to(device)
        time_in = torch.tensor([t], dtype = torch.float32).to(device)

        epsilon = denoiser(X_input, time_in).numpy(force=True)
         
        epsilon_classifier = calc_gradients(obstacle_center, X_t, client_id, panda, panda_joints)
        cl_weightage = (t/T) * 0.1 # np.clip((1 - t/T), 0.01, 1) * 0.1
        
        X_t = (1/np.sqrt(alpha[t-1])) * (X_t - ((1 - alpha[t-1])/(np.sqrt(1 - alpha_bar[t-1]))) * epsilon) + beta[t-1]*z + cl_weightage*epsilon_classifier
        
        # CONDITIONING:
        X_t[:, :, 0] = q0 # start[:] # rest_poses # start[:]
        X_t[:, :, -1] = target # goal[:]
        
        reverse_diff_traj[t-1] = einops.rearrange(X_t[0], 'c n -> n c').copy()

    print("Done!!!")
    return reverse_diff_traj[0]


if __name__=='__main__':
    for i in range(2, 12):
        path_to_dataset = '/home/vishal/Volume_E/Active/Undergrad_research/CoRL2023/datasets/nvidia_MPI/mpinets_hybrid_training_data/val/val.hdf5'
        f = h5py.File(path_to_dataset, 'r')
        traj = f['hybrid_solutions'][i, :, :]
        start_traj = traj[0, :]
        end_traj = traj[-1, :]

        gen_trajectory(conditioning=True, start=start_traj, goal=end_traj, traj_count=i)
        print("")


