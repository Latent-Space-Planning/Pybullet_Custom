import torch
import time
import os
import matplotlib.pyplot as plt
import einops
import numpy as np
import pybullet as p

import sys 

sys.path.append('/home/jayaram/research/research_tracks/table_top_rearragement/Pybullet_Custom')
# sys.path.append('/home/jayaram/research/research_tracks/table_top_rearragement/Pybullet_Custom/diffusion/priors')
# sys.path.append('/home/jayaram/research/research_tracks/table_top_rearragement/Pybullet_Custom/diffusion/scripts')

from diffusion.scripts.diffusion_functions import *
from diffusion.priors.bezier_curve_2d import *
from diffusion.Models.temporalunet import TemporalUNet

from train_SDF import SDFModel_min_dist_per_joint, SDFModel_min_dist_overall, SDFModel
import h5py

from torch_forward_kinematics_panda import Torch_fwd_kinematics_Panda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model_name = "/home/jayaram/research/research_tracks/table_top_rearragement/global_classifier_guidance_for_7DOF_manipulator/diffuser_ckpts_7dof_mpinets/7dof/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)

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

    np.save(f"/home/jayaram/research/research_tracks/table_top_rearragement/Pybullet_Custom/results/7dof/generated_trajectories/reverse_trajectory_st_gl_fix_val_{traj_count}.npy", reverse_diff_traj)

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

def clip_joints_torch(q):
    lower_lims = torch.tensor([-2.8973000049591064, -1.7627999782562256, -2.8973000049591064, -3.0717999935150146, -2.8973000049591064, -0.017500000074505806, -2.8973000049591064])
    upper_lims = torch.tensor([2.8973000049591064, 1.7627999782562256, 2.8973000049591064, -0.0697999969124794, 2.8973000049591064, 3.752500057220459, 2.8973000049591064])
    return torch.clip(q.T, lower_lims, upper_lims)

def value_func_and_grad(obstacle_center, link_pos):
    '''
    Returns:
    Value: Distance to the obstacle
    Grad: Gradient of Value w.r.t. pose of the link
    '''
    dist = np.linalg.norm(obstacle_center - link_pos) # **2
    if dist > 0.3:
        grad = np.zeros_like(link_pos) # np.exp(-2*(link_pos - obstacle_center)) # 
    else:
        grad = 2*(link_pos - obstacle_center)
    # grad[2] = 0
    return dist, grad.reshape((3, 1))

def value_func_and_grad_torch(obstacle_centers, link_pos):
    '''
    Returns:
    Value: Distance to the obstacle
    Grad: Gradient of Value w.r.t. pose of the link
    '''
    # obstacle_centers = torch.tensor(np.array(obstacle_centers)) # Shape: num_obstacles*3 
    # dist = torch.log(torch.norm(obstacle_center - link_pos)+ 1)  # **2
    net_dist = torch.scalar_tensor(0)#, requires_grad=True)
    grad = torch.zeros_like(link_pos)
    # st_time = time.time()
    for obs_cen in obstacle_centers:
        net_dist += torch.sum(torch.clip(torch.norm(obs_cen - link_pos, dim=1), max=0.8))#torch.min(torch.norm(obs_cen - link_pos), torch.ones_like(link_pos)*0.3, keepdim=True))
    # print("Net distance: {}")
    # print(f"Time check for all obstacles: {time.t}")

    
    # collision_score = torch.where(dist > 0.4, dist, torch.tensor(0.0))

    # if dist > 0.3:
    #     collision_score = torch.norm(obstacle_center - link_pos)
    #     grad = torch.zeros_like(link_pos)
    # else:
    #     collision_score = torch.scalar_tensor(0.3)
    #     grad = 2 * (link_pos - obstacle_center)
    # grad[2] = 0
    return net_dist, grad #.view(1, 3, 1)

def calc_gradients_batch_torch_kinematics(obstacle_centers, traj_batch, client_id, panda, panda_joints, fwd_kinematics):
    # is this even possible? (How do we resetJointstate for a batch of configurations in a single simulator?) : No need to do that as we already have trajectory
    batch_size = traj_batch.shape[0]    #B
    num_joints = traj_batch.shape[1]
    traj_len = traj_batch.shape[2]

    gradients = np.zeros(traj_batch.shape, dtype=float)  # Shape: (B, 7, traj_len)

    # traj_batch shape: (25, 7, 50)
    traj_batch_lin = traj_batch.permute(1, 0, 2).reshape(7, -1)    # combine conf of all trajs in current batch  (7, B*traj_len)

    # st_time = time.time()
    clipped_q_batch = clip_joints_torch(traj_batch_lin)
    clipped_q_batch = torch.concatenate([clipped_q_batch, torch.zeros(batch_size*traj_len, 2)], dim=1)  # : (B*traj_len = 1250, 9)
    # print(f"Clip Joints: {time.time() - st_time}")

    # st_time = time.time()
    link_poses_batch = fwd_kinematics.get_batch_differential_forward_kinematics(clipped_q_batch)
    # print(f"Forward kine time: {time.time() - st_time}")

    # st_time = time.time()
    coll_score = 0.0
    for b in range(traj_batch.shape[0]):    #can we compute score by stacking all the traj batches and backpropogate , I dont think so (we need to iterate over each traj_batch seperately)
        for l_k in link_poses_batch.keys():
            coll_score += value_func_and_grad_torch(obstacle_centers, link_poses_batch[l_k][:, :3, 3])[0]
        # print(f"Collision Check time for link {l_k}: {time.time() - st_time2}")
    # print(f"Collision Check time for all links: {time.time() - st_time}")

    # st_time = time.time()
    coll_score.backward()
    # print(f"Backward time: {time.time() - st_time}")

    return True

    # for j in range(traj_len):
    #     q_batch = traj_batch[:, :, j]    #jth conf in batch of trajs (traj_batch)
    #     # clipped_q_batch = clip_joints(q_batch)
        
    #     clipped_q_batch = clip_joints_torch(q_batch)
    #     clipped_q_batch = torch.concatenate([clipped_q_batch, torch.zeros(batch_size, 2)], dim=1)  # : (B, 9)
    #     print(f"Clip Joints: {time.time() - st_time}")

    #     link_poses_batch = fwd_kinematics.get_batch_differential_forward_kinematics(clipped_q_batch)    #dict : link positions i.w world2link for all 10 links in current batch of conf)
    #     # com_trn_batch = client_id.getLinkState(panda, 6, computeForwardKinematics=1)[2]
    #     # J_t_batch, J_r_batch = client_id.calculateJacobian(panda, 6, list(com_trn_batch), clipped_q_batch.tolist(), np.zeros_like(q_batch).tolist(), np.zeros_like(q_batch).tolist())
    #     # J_t_batch = np.array(J_t_batch)
    #     # J_r_batch = np.array(J_r_batch)

    #     # we can directly compute clipped q grad from torch FK
    #     st_time = time.time()
    #     coll_score = 0.0
    #     for link in link_poses_batch.keys():
    #         link_pos = link_poses_batch[link][:, :3, 3]    #(B, 3)
    #         #  get collision score (sum up the distances of current link pos from all obstacles)
    #         for k in range(len(obstacle_centers)):
    #             coll_score += value_func_and_grad_torch(obstacle_centers[k], link_pos)[0]
    #         # link_pos.norm().backward()
    #     print(f"Value function time: {time.time() - st_time}")

    #     # print('coll score for traj j : {}'.format(coll_score))
    #     st_time = time.time()
    #     coll_score.backward()
    #     print(f"Backward time: {time.time() - st_time}")

    #     # grad_batch = None
    #     # for k in range(len(obstacle_centers)):
    #     #     if k == 0:
    #     #         values, grad_batch = value_func_and_grad_batch(obstacle_centers[k], link_pos_batch)
    #     #     else:
    #     #         _, temp_grad_batch = value_func_and_grad_batch(obstacle_centers[k], link_pos_batch)
    #     #         grad_batch += temp_grad_batch

    #     # grad_batch = grad_batch / len(obstacle_centers)
    #     # net_grad_batch = np.sum(grad_batch[:, :, np.newaxis] * J_t_batch[:, :7], axis=1)
    #     # denom_batch = np.linalg.norm(net_grad_batch, axis=1)

    #     # nonzero_denom_indices = denom_batch > 0
    #     # net_grad_batch[nonzero_denom_indices] = net_grad_batch[nonzero_denom_indices] / denom_batch[nonzero_denom_indices, np.newaxis]

    #     # gradients[:, :, j] = net_grad_batch / num_joints

    # return np.clip(gradients, -1, 1)

def calc_gradients_batch_brute_force(obstacle_centers, traj_batch, client_id, panda, panda_joints):
    batch_size = traj_batch.shape[0]
    num_joints = traj_batch.shape[1]
    traj_len = traj_batch.shape[2]

    gradients = np.zeros(traj_batch.shape, dtype=float)  # Shape: (B, 7, traj_len)

    for j in range(traj_len):
        for b in range(batch_size):
            q = traj_batch[b, :, j]
            clipped_q = clip_joints(q)

            for i, joint_ind in enumerate(panda_joints):
                client_id.resetJointState(panda, joint_ind, clipped_q[i])

            link_pos = client_id.getLinkState(panda, 6, computeForwardKinematics=1)[0]
            com_trn = client_id.getLinkState(panda, 6, computeForwardKinematics=1)[2]
            J_t, J_r = client_id.calculateJacobian(panda, 6, list(com_trn), clipped_q.tolist(), np.zeros_like(q).tolist(), np.zeros_like(q).tolist())
            J_t = np.array(J_t)
            J_r = np.array(J_r)

            grad = None
            for k in range(len(obstacle_centers)):
                if k == 0:
                    value, grad = value_func_and_grad(obstacle_centers[k], np.array(link_pos))
                else:
                    _, temp = value_func_and_grad(obstacle_centers[k], np.array(link_pos))
                    grad += temp

            grad = grad / len(obstacle_centers)
            net_grad = np.sum(grad * J_t[:, :7], axis=0)
            denom = np.linalg.norm(net_grad)

            if denom > 0:
                net_grad = net_grad / denom

            gradients[b, :, j] += net_grad / num_joints

    return np.clip(gradients, -1, 1)

    # # Example batch of trajectories (replace with your data)
    # batch_size = 16
    # traj_batch = np.random.random((batch_size, 7, 50))

    # # Call the function with the batch of trajectories
    # obstacle_centers = []  # Replace with actual obstacle centers
    # result = calc_gradients_batch(obstacle_centers, traj_batch, client_id, panda, panda_joints)
    # print(result.shape)  # Shape: (batch_size, 7, traj_len)

def calc_gradients(obstacle_centers, traj, client_id, panda, panda_joints):
    '''Calculate Gradients
    '''
    gradients = np.zeros(traj.shape, dtype=float)   #(1, 7, 50)
    for j in range(50):   #loop thru each conf in traj
        numJoints = p.getNumJoints(panda)
        # q = np.zeros(shape=(9, )) # + 0.02
        # q[:7] = traj[0, :, j].reshape((7, )) # get jth conf in current traj
        # # print(q)
        # # mpos, mvel, mtorq = getMotorJointStates(panda)
        # # q = q0 = np.array([-0.5778415 ,  0.54352427,  0.6913696 , -2.5508945 , -0.11496189, 3.0910137, -2.49674   ])
        # # numJoints = p.getNumJoints(panda)
        # # ee_ind = numJoints - 1
        # # print(ee_ind)
        # # print(f'Panda id: {panda}')
        # # print(np.array([0, 0, 0], dtype=float), q, np.zeros_like(q), np.zeros_like(q))
        # clipped_q = np.zeros_like(q)   #clip joints
        # clipped_q[:7] = clip_joints(q[:7])
        
        # for jj, joint_ind in enumerate(panda_joints):   #previously it was i here (its wrong)
        #     client_id.resetJointState(panda, joint_ind, clipped_q[jj])

        for i in range(12):# numJoints):     #loop thru each link
            q = np.zeros(shape=(9, )) # + 0.02
            q[:7] = traj[0, :, j].reshape((7, )) # get jth conf in current traj
            # # print(q)
            # # mpos, mvel, mtorq = getMotorJointStates(panda)
            # # q = q0 = np.array([-0.5778415 ,  0.54352427,  0.6913696 , -2.5508945 , -0.11496189, 3.0910137, -2.49674   ])
            # # numJoints = p.getNumJoints(panda)
            # # ee_ind = numJoints - 1
            # # print(ee_ind)
            # # print(f'Panda id: {panda}')
            # # print(np.array([0, 0, 0], dtype=float), q, np.zeros_like(q), np.zeros_like(q))
            clipped_q = np.zeros_like(q)   #clip joints
            clipped_q[:7] = clip_joints(q[:7])
            
            for jj, joint_ind in enumerate(panda_joints):   #previously it was i here (its wrong)
                client_id.resetJointState(panda, joint_ind, clipped_q[jj])

            # print('i :{}'.format(i))

            link_pos = client_id.getLinkState(panda, i, computeForwardKinematics=1)[0]   #i is always 6 here, bug?
            com_trn = client_id.getLinkState(panda, i, computeForwardKinematics=1)[2]
            J_t, J_r = client_id.calculateJacobian(panda, i, list(com_trn), clipped_q.tolist(), np.zeros_like(q).tolist(), np.zeros_like(q).tolist())
            J_t = np.array(J_t)   #(3, 9)
            J_r = np.array(J_r)   #(3, 9)

            grad = None   
            for k in range(len(obstacle_centers)):
                if k==0:
                    value, grad = value_func_and_grad(obstacle_centers[k], np.array(link_pos))# J_t@clipped_q.reshape((9, 1)))
                else:
                    _, temp = value_func_and_grad(obstacle_centers[k], np.array(link_pos))
                    grad += temp  #(3, 1)
            grad = grad/len(obstacle_centers)
            # if np.linalg.norm(grad) > 0:
            #     print(f"Value: {value}, Grad: {grad}")
            # print("HIIIIIIIIIII")
            net_grad = np.sum(grad * J_t[:, :7], axis=0) # .reshape((traj[0, :, j].shape[0], traj[0, :, j].shape[1]))
            denom = np.linalg.norm(net_grad)
            if denom > 0:
                net_grad = net_grad / denom    #(7, )

            gradients[0, :, j] += net_grad.reshape((traj[0, :, j].shape))/7
    # print(f"Gradients: {gradients}")
    print(f"Norm of gradient: {np.linalg.norm(gradients)}\t Max gradient: {np.max(gradients)}")
    return np.clip(gradients, -1, 1)    #(1, 7, 50)

def calc_gradients_from_sdf_network(obstacle_centers, traj, client_id, panda, panda_joints):
    '''Calculate Gradients
    '''
    gradients = np.zeros(traj.shape, dtype=float)   #(1, 7, 50)
    traj = einops.rearrange(traj[0], 'c n -> n c').copy()
    traj = torch.tensor(traj, dtype = torch.float32).to(device="cuda")
    traj.requires_grad = True

    checkpoint_path = "/home/jayaram/research/research_tracks/table_top_rearragement/Pybullet_Custom/logs/sdf_single_min_distances_per_link/state_199.pt"
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    # Access the desired variables from the checkpoint
    model_state_dict = checkpoint['model']
    # optimizer_state_dict = checkpoint['optimizer']
    # epoch = checkpoint['epoch']

    # Create the model and optimizer objects
    # change action dim to 2 later
    input_dim = 7    # no of joints  (7)
    hidden_dim_1 =  32  #
    hidden_dim_2 = 128
    hidden_dim_3 = 64 
    # output_dim = 21   # 7*3
    output_dim = 10    # 10*3
    #load model architecture 
    model = SDFModel_min_dist_per_joint(input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim)
    model = model.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # Load the model and optimizer states from the checkpoint
    model.load_state_dict(model_state_dict)
    
    for param in model.parameters():
        param.requires_grad = False

    # gt_dists = np.zeros((50, 30), dtype=float)   #(1, 7, 50)
    # for conf_no in range(50):   #loop thru each conf in traj
    #     for l, joint_ind in enumerate(panda_joints):
    #         client_id.resetJointState(panda, joint_ind, traj[conf_no, l])
    #     for link_no in range(10):# numJoints):
    #         val = 100.0 * opengjk.pygjk(link_vertices[conf_no, link_no], obstacles_bbox_vertices[obstacle_no])
    #         # print(value)
    #         if(val > 0.0): 
    #             dists[conf_no, link_no, obstacle_no] = val
    #         else: 
    #             # print('dist is 0.0, so compute volume')
    #             dists[conf_no, link_no, obstacle_no] = -100000.0 * cuboid_overlap_volume2(link_vertices[conf_no, link_no], obstacles_bbox_vertices[obstacle_no])

    dists = model(traj)   #check if this is the gt   --  (50, 21)

    gradients = torch.autograd.grad(torch.sum(dists), traj)[0].detach().cpu()
    gradients = gradients.numpy()
    gradients = gradients[np.newaxis, :] 
    gradients = einops.rearrange(gradients, 'b c n -> b n c').copy()

    # print(f"Gradients: {gradients}")
    print(f"Norm of gradient: {np.linalg.norm(gradients)}\t Max gradient: {np.max(gradients)}")
    # scale gradients to [-1 to 1] based on min and max values 
    min_value_abs = np.abs(np.min(gradients))
    max_value = np.max(gradients)

    # Divide negative values by the minimum value and positive values by the maximum value
    clipped_grads = np.where(gradients < 0, gradients / min_value_abs, gradients / max_value)

    return clipped_grads
    # return np.clip(gradients, -1, 1)    #(1, 7, 50)


def infer_guided_batch(denoiser, q0, target, obstacle_centers, client_id, panda, panda_joints):
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
    T = 255 #255 # 5
    batch_size = 25

    beta = schedule_variance(T)
    alpha = 1 - beta
    alpha_bar = np.array(list(map(lambda t:np.prod(alpha[:t]), np.arange(T+1)[1:])))

    mean = np.zeros(traj_len)
    cov = np.eye(traj_len)

    obstacle_centers = [torch.tensor(arr) for arr in obstacle_centers]   #comment this when using calc_gradients fn 

    xT_batch = np.random.multivariate_normal(mean, cov, (batch_size, 7))    #(B, 7, 50)
    X_t_batch = xT_batch.copy()
    xT_batch = torch.tensor(xT_batch, dtype=torch.float, requires_grad=True)   #comment this when using calc_gradients fn  

    # CONDITIONING:
    rest_poses = np.array([-0.00029003781167199756, -0.10325848749542864, 0.00020955647419784322,
                                    -3.0586072742527284, 1.3219639494062742e-05, 2.9560190438374425, 2.356101763339709])
    X_t_batch[:, :, 0] = q0 # start[:] # start[:]   -- broadcasting
    X_t_batch[:, :, -1] = target # goal[:]

    reverse_diff_traj_batch = np.zeros((T+1, batch_size, traj_len, 7))   #len -- 256 -- T+1   #(T+1, B, 50, 7)
    # reverse_diff_traj_batch[T] = einops.rearrange(np.array(xT_batch), 'b c n ->b n c').copy()
    reverse_diff_traj_batch[T] = einops.rearrange(X_t_batch, 'b c n ->b n c').copy()

    denoiser.train(False)
    fwd_kinematics = Torch_fwd_kinematics_Panda()   #comment this when using calc_gradients fn  

    # st_time = time.time()
    # print("bo")
    for t in range(T, 0, -1):
        # print("yo")
        # xT_batch.requires_grad = True     #comment this when using calc_gradients fn  
        print(f"\rTimestep: {t}", end="")

        if t > 1:
            z = np.random.multivariate_normal(mean, cov, (batch_size, 7))   #(B, 7, 50)
        else:
            z = np.zeros((batch_size, 7, traj_len))
        
        X_input_batch = torch.tensor(X_t_batch, dtype = torch.float).to(device)   #(B,7,50)
        time_in_batch = torch.tensor([t] * batch_size, dtype = torch.float).to(device)

        epsilon = denoiser(X_input_batch, time_in_batch).numpy(force=True)   #(B, 7, 50)
        # st_time = time.time()
        # epsilon_classifier = calc_gradients(obstacle_centers, X_t_batch, client_id, panda, panda_joints)
        # epsilon_classifier = calc_gradients_batch_torch_kinematics(obstacle_centers, X_t_batch, client_id, panda, panda_joints)
        calc_gradients_batch_torch_kinematics(obstacle_centers, xT_batch, client_id, panda, panda_joints, fwd_kinematics)
        # print(f"calc grad total time: {time.time() - st_time}")


        epsilon_classifier = torch.clip(xT_batch.grad, -1, 1)       #comment this when using calc_gradients fn  
        # Most of the elements are zero here. Shouldnt be.

        # cl_weightage = np.clip(np.log(1+((t-2)/T)*(np.exp(1) - 1)), 0.005, 1) # np.clip((t/T) * 0.1, 0.001, 1) # np.clip((1 - t/T), 0.01, 1) * 0.1
        if t>=7:    # t = 7 is working well with np.clip((t/T) * 0.01, 0.001, 1)
            cl_weightage = np.clip((t/T) * 0.01, 0.001, 1)
        else:
            cl_weightage = 0.001   #0.00001

        # cl_weightage = 100
        X_t_batch = (1/np.sqrt(alpha[t-1])) * (X_t_batch - ((1 - alpha[t-1])/(np.sqrt(1 - alpha_bar[t-1]))) * epsilon) + beta[t-1]*z + cl_weightage*epsilon_classifier.numpy()
        
        # CONDITIONING:
        X_t_batch[:, :, 0] = q0 # start[:] # rest_poses # start[:]  -- broadcasting
        X_t_batch[:, :, -1] = target # goal[:]
        
        reverse_diff_traj_batch[t-1] = einops.rearrange(X_t_batch, 'b c n -> b n c').copy()
        
         #comment below when using calc_gradients fn  
        xT_batch.detach()
        xT_batch = torch.Tensor(X_t_batch)

    print("Done!!!")
    return reverse_diff_traj_batch[0]

def infer_guided(denoiser, q0, target, obstacle_centers, client_id, panda, panda_joints):
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
    X_t = xT.copy()   #(1, 7, 50)

    # CONDITIONING:
    rest_poses = np.array([-0.00029003781167199756, -0.10325848749542864, 0.00020955647419784322,
                                    -3.0586072742527284, 1.3219639494062742e-05, 2.9560190438374425, 2.356101763339709])
    X_t[:, :, 0] = q0 # start[:] # start[:]
    X_t[:, :, -1] = target # goal[:]

    reverse_diff_traj = np.zeros((T+1, traj_len, 7))   #len -- 256 -- T+1   #(T+1, 50, 7)
    reverse_diff_traj[T] = einops.rearrange(xT[0], 'c n -> n c').copy()

    denoiser.train(False)

    for t in range(T, 0, -1):
        print(f"\rTimestep: {t}", end="")

        if t > 1:
            z = np.random.multivariate_normal(mean, cov, (1, 7))   #(1, 7, 50)
        else:
            z = np.zeros((1, 7, traj_len))
        
        X_input = torch.tensor(X_t, dtype = torch.float32).to(device)   #(1,7,50)
        time_in = torch.tensor([t], dtype = torch.float32).to(device)

        epsilon = denoiser(X_input, time_in).numpy(force=True)   #(1, 7, 50)
         
        # epsilon_classifier = calc_gradients_from_sdf_network(obstacle_centers, X_t, client_id, panda, panda_joints)
        epsilon_classifier = calc_gradients(obstacle_centers, X_t, client_id, panda, panda_joints)
        # cl_weightage = np.clip(np.log(1+((t-2)/T)*(np.exp(1) - 1)), 0.005, 1) # np.clip((t/T) * 0.1, 0.001, 1) # np.clip((1 - t/T), 0.01, 1) * 0.1
        
        if(t < 10):    #gudiance weightage should be higher for lower time steps (from 10 -150) as at later timesteps, traj are noisy, so gudiance doesnt make much sense unless we use mechanism to denoise to x0 and apply classifier gudiance
            cl_weightage = 0.0001   #here unconditional denoiser should be reposnsible to make traj smoother a very low time steps
        else: 
            cl_weightage = np.log(1+((t-2)/T)*(np.exp(1) - 1))     # reduces from 0.99 to 0.05

        X_t = (1/np.sqrt(alpha[t-1])) * (X_t - ((1 - alpha[t-1])/(np.sqrt(1 - alpha_bar[t-1]))) * epsilon) + beta[t-1]*z + cl_weightage*epsilon_classifier
        
        # CONDITIONING:
        X_t[:, :, 0] = q0 # start[:] # rest_poses # start[:]
        X_t[:, :, -1] = target # goal[:]
        
        reverse_diff_traj[t-1] = einops.rearrange(X_t[0], 'c n -> n c').copy()

    print("Done!!!")
    return reverse_diff_traj[0]


if __name__=='__main__':
    for i in range(2, 12):
        path_to_dataset = '/home/jayaram/research/research_tracks/table_top_rearragement/global_classifier_guidance_for_7DOF_manipulator/datasets/mpinets_hybrid_training_data/val/val.hdf5'
        f = h5py.File(path_to_dataset, 'r')
        traj = f['hybrid_solutions'][i, :, :]
        start_traj = traj[0, :]
        end_traj = traj[-1, :]

        gen_trajectory(conditioning=True, start=start_traj, goal=end_traj, traj_count=i)
        print("")


