import numpy as np
import torch
import os
import re
from einops.layers.torch import Rearrange
import pybullet as p

class IntersectionVolumeGuide:

    def __init__(self, env, obstacle_config, device, clearance):

        self.env = env
        self.device = device
        self.clearance = clearance
        self.obstacle_config = np.array(obstacle_config)
        self.obs_ids = []

        self.static_dh_params = torch.tensor([[0, 0.333, 0, 0],
                                  [0, 0, -torch.pi / 2, 0],
                                  [0, 0.316, torch.pi / 2, 0],
                                  [0.0825, 0, torch.pi / 2, 0],
                                  [-0.0825, 0.384, -torch.pi / 2, 0],
                                  [0, 0, torch.pi / 2, 0],
                                  [0.088, 0, torch.pi / 2, 0],
                                  [0, 0.107, 0, 0],
                                  [0, 0, 0, -torch.pi / 4],
                                  [0.0, 0.1034, 0, 0]], dtype = torch.float32, device = self.device)
        
        self.define_link_information()
        # self.define_obstacles(obstacle_config)

        self.rearrange_joints = Rearrange('batch channels traj_len -> batch traj_len channels')
    
    def get_tf_mat(self, dh_params):
        
        # dh_params is (batch, traj_len, 1, 4)
        a = dh_params[:, :, 0]
        d = dh_params[:, :, 1]
        alpha = dh_params[:, :, 2]
        q = dh_params[:, :, 3]

        transform = torch.zeros(dh_params.shape[0], dh_params.shape[1], 4, 4)

        transform[:, :, 0, 0] = torch.cos(q)
        transform[:, :, 0, 1] = -torch.sin(q)
        transform[:, :, 0, 3] = a

        transform[:, :, 1, 0] = torch.sin(q) * torch.cos(alpha)
        transform[:, :, 1, 1] = torch.cos(q) * torch.cos(alpha)
        transform[:, :, 1, 2] = -torch.sin(alpha)
        transform[:, :, 1, 3] = -torch.sin(alpha) * d

        transform[:, :, 2, 0] = torch.sin(q) * torch.sin(alpha)
        transform[:, :, 2, 1] = torch.cos(q) * torch.sin(alpha)
        transform[:, :, 2, 2] = torch.cos(alpha)
        transform[:, :, 2, 3] = torch.cos(alpha) * d

        transform[:, :, -1, -1] = 1

        # Return transform of (batch, traj_len, 4, 4)
        return transform
    
    def forward_kinematics(self, joints):
        """
        joint_angles => Array of shape (batch, 7, traj_len)
        """        

        dh_params = torch.clone(self.static_dh_params)
        dh_params = dh_params.unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1)
        dh_params[:, :, :7, 3] = joints[:, :, :]
        # dh_params is (batch, traj_len, 10, 4)

        fk = torch.zeros(size = (dh_params.shape[0], dh_params.shape[1], 9, 4, 4), dtype = torch.float32, device = self.device)
        # fk is (batch, traj_len, 9, 4, 4)
    
        T = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1)
        for i in range(7):
            dh_matrix = self.get_tf_mat(dh_params[:, :, i, :]) 
            # T is (batch, traj_len, 4, 4)
            # dh_matrix is (batch, traj_len, 4, 4)
            T = torch.matmul(T, dh_matrix) 
            if i == 6:
                fk[:, :, i:, :, :] = T.unsqueeze(2)
            else:
                fk[:, :, i, :, :] = T

        return fk
    
    def get_end_effector_transform(self, joints):

        # joints is (1, 50, 7) tensor

        dh_params = torch.clone(self.static_dh_params)
        dh_params = dh_params.unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1)
        dh_params[:, :, :7, 3] = joints[:, :, :]
        # dh_params is (1, 50, 10, 4)

        T = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1)
        for i in range(10):
            dh_matrix = self.get_tf_mat(dh_params[:, :, i, :]) 
            # T is (batch, traj_len, 4, 4)
            # dh_matrix is (batch, traj_len, 4, 4)
            T = torch.matmul(T, dh_matrix)

        return T        

    def define_obstacles(self, obstacle_config, t):

        # Obstacle Config => (n, 10)
        
        obstacle_sizes = np.array(obstacle_config[:, 7:])
        if t > 20:
            obstacle_sizes = np.maximum(obstacle_sizes, 0.2)

        if t != 0:
            obstacle_sizes = torch.tensor(obstacle_sizes, dtype = torch.float32, device = self.device) + self.clearance[t]
        else:
            obstacle_sizes = torch.tensor(obstacle_sizes, dtype = torch.float32, device = self.device)
        
        obstacle_static_vertices = self.get_vertices(obstacle_sizes)
        # obstacle_static_vertices => (n, 4, 8)

        obstacle_transform = np.zeros((obstacle_config.shape[0], 4, 4))
        for i in range(obstacle_config.shape[0]):
            obstacle_transform[i, :3, :3] = np.array(self.env.client_id.getMatrixFromQuaternion(obstacle_config[i, 3:7])).reshape((3, 3))
            obstacle_transform[i, :3, -1] = obstacle_config[i, :3]
        obstacle_transform[:, -1, -1] = 1.
        # obstacle_transform => (n, 4, 4)

        obstacle_transform = torch.tensor(obstacle_transform, dtype = torch.float32, device = self.device)
        obstacle_vertices = torch.bmm(obstacle_transform, obstacle_static_vertices)
        # obstacle_vertices => (n, 4, 8)

        self.obs_min = torch.min(obstacle_vertices, dim = -1)[0][:, :-1]
        self.obs_max = torch.max(obstacle_vertices, dim = -1)[0][:, :-1]
        # obs_min => (n, 3)
        # obs_max => (n, 3)
    
    def get_vertices(self, dimensions):

        l, b, h = dimensions[:, 0], dimensions[:, 1], dimensions[:, 2]
        obstacle_vertices = torch.zeros(size = (dimensions.shape[0], 4, 8), dtype = torch.float32, device = self.device)

        obstacle_vertices[:, 0, 0] = -l/2
        obstacle_vertices[:, 0, 1] = l/2
        obstacle_vertices[:, 0, 2] = l/2
        obstacle_vertices[:, 0, 3] = -l/2
        obstacle_vertices[:, 0, 4] = -l/2
        obstacle_vertices[:, 0, 5] = l/2
        obstacle_vertices[:, 0, 6] = l/2
        obstacle_vertices[:, 0, 7] = -l/2

        obstacle_vertices[:, 1, 0] = -b/2
        obstacle_vertices[:, 1, 1] = -b/2
        obstacle_vertices[:, 1, 2] = b/2
        obstacle_vertices[:, 1, 3] = b/2
        obstacle_vertices[:, 1, 4] = -b/2
        obstacle_vertices[:, 1, 5] = -b/2
        obstacle_vertices[:, 1, 6] = b/2
        obstacle_vertices[:, 1, 7] = b/2

        obstacle_vertices[:, 2, 0] = -h/2
        obstacle_vertices[:, 2, 1] = -h/2
        obstacle_vertices[:, 2, 2] = -h/2
        obstacle_vertices[:, 2, 3] = -h/2
        obstacle_vertices[:, 2, 4] = h/2
        obstacle_vertices[:, 2, 5] = h/2
        obstacle_vertices[:, 2, 6] = h/2
        obstacle_vertices[:, 2, 7] = h/2

        obstacle_vertices[:, 3, :] = 1.
        
        return obstacle_vertices
    
    def define_link_information(self):

        links_folder_path = '/home/failedmesh/miniconda3/lib/python3.10/site-packages/pybullet_data/franka_panda/meshes/collision/'
        try:
            link_file_names = os.listdir(links_folder_path)
        except OSError as e:
            print(f"Error reading files in folder: {e}")

        self.link_index_to_name = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
                                   'hand', 'finger']
        self.link_dimensions_from_name = {}
        
        for file_name in link_file_names:

            if file_name[-4:] == ".obj":
                vertices = []    
                link_name = file_name[:-4]
                with open(links_folder_path + file_name, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('v '):
                            vertex = re.split(r'\s+', line)[1:4]
                            vertex = np.array([float(coord) for coord in vertex])
                            vertices.append(vertex)
                max_point = np.max(np.array(vertices), axis = 0)
                min_point = np.min(np.array(vertices), axis = 0)
                self.link_dimensions_from_name[link_name] = max_point - min_point

        self.link_dimensions = []
        
        for link_index in range(len(self.link_index_to_name)):

            link_name = self.link_index_to_name[link_index]
            link_dimensions = self.link_dimensions_from_name[link_name].copy()
            
            if link_index == len(self.link_index_to_name) - 1:
                link_dimensions[1] *= 4
            
            self.link_dimensions.append(link_dimensions)
        self.link_dimensions = torch.tensor(np.array(self.link_dimensions), dtype = torch.float32, device = self.device)

        self.link_vertices = self.get_vertices(self.link_dimensions)

        self.link_static_joint_frame = [1, 2, 3, 4, 5, 6, 7, 7, 7] 
        self.static_frames = []

        # Link 0:
        self.static_frames.append([[1., 0., 0., 8.71e-05],
                                   [0., 1., 0., -3.709035e-02],
                                   [0., 0., 1., -6.851545e-02],
                                   [0., 0., 0., 1.]])
        # Link 1:
        self.static_frames.append([[1., 0., 0., -8.425e-05],
                                   [0., 1., 0., -6.93950016e-02],
                                   [0., 0., 1., 3.71961970e-02],
                                   [0., 0., 0., 1.]])
        
        # Link 2:
        self.static_frames.append([[1., 0., 0., 0.0414576],
                                   [0., 1., 0., 0.0281429],
                                   [0., 0., 1., -0.03293086],
                                   [0., 0., 0., 1.]])

        # Link 3:
        self.static_frames.append([[1., 0., 0., -4.12337575e-02],
                                   [0., 1., 0., 3.44296512e-02],
                                   [0., 0., 1., 2.79226985e-02],
                                   [0., 0., 0., 1.]])

        # Link 4:
        self.static_frames.append([[1., 0., 0., 3.3450000e-05],
                                   [0., 1., 0., 3.7388050e-02],
                                   [0., 0., 1., -1.0619285e-01],
                                   [0., 0., 0., 1.]])

        # Link 5:
        self.static_frames.append([[1., 0., 0., 4.21935000e-02],
                                   [0., 1., 0., 1.52195003e-02],
                                   [0., 0., 1., 6.07699933e-03],
                                   [0., 0., 0., 1.]])

        # Link 6:
        self.static_frames.append([[1., 0., 0., 1.86357500e-02],
                                   [0., 1., 0., 1.85788569e-02],
                                   [0., 0., 1., 7.94137484e-02],
                                   [0., 0., 0., 1.]])

        # Link 7:
        self.static_frames.append([[7.07106767e-01, 7.07106795e-01, 0., -1.26717073e-03],
                                   [-7.07106795e-01, 7.07106767e-01, 0., -1.25294673e-03],
                                   [0., 0., 1., 1.27018693e-01],
                                   [0., 0., 0., 1.]])

        # Link 8:
        self.static_frames.append([[7.07106767e-01, 7.07106795e-01, 0., 9.29352476e-03],
                                   [-7.07106795e-01, 7.07106767e-01, 0., 9.28272434e-03],
                                   [0., 0., 1., 1.92390375e-01],
                                   [0., 0., 0., 1.]])
        
        self.static_frames = torch.tensor(self.static_frames, dtype = torch.float32, device = self.device)
        
    def get_link_transform(self, joints):

        joint_transform = self.forward_kinematics(joints)

        # joint_transform => (batch, traj_len, 9, 4, 4)
        # static_frames => (9, 4, 4)
        link_transform = joint_transform @ self.static_frames.unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1, 1)

        return link_transform
    
    def cost(self, joint_input, t):

        # joint_angles => (b, 7, n)
        # obs_min => (n, 3)
        # obs_max => (n, 3) 
        self.define_obstacles(self.obstacle_config, t)

        joints = self.rearrange_joints(joint_input)
        # Now joints is (batch, traj_len, 7)
        
        link_transform = self.get_link_transform(joints)

        b = link_transform.shape[0]
        n = link_transform.shape[1]
        nl = link_transform.shape[2]
        no = self.obs_min.shape[0]

        # link_transform => (batch, traj_len, 9, 4, 4)
        # self.link_vertices => (9, 4, 8)
        link_vertices = link_transform @ self.link_vertices.unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1, 1)
        link_vertices = link_vertices[:, :, :, :3, :]

        # link_vertices => (batch, traj_len, 9, 3, 8)
        link_min = torch.min(link_vertices, dim = -1)[0]
        link_max = torch.max(link_vertices, dim = -1)[0]

        # link_min => (batch, traj_len, 9, 3)
        # link_max => (batch, traj_len, 9, 3)
        expanded_link_min = link_min.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n, no*nl, 3)
        expanded_link_max = link_max.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n, no*nl, 3)

        expanded_obs_min = self.obs_min.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, n, nl, 1, 1).view(b, n, no*nl, 3)
        expanded_obs_max = self.obs_max.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, n, nl, 1, 1).view(b, n, no*nl, 3)

        overlap_min = torch.max(expanded_link_min, expanded_obs_min)
        overlap_max = torch.min(expanded_link_max, expanded_obs_max)

        overlap_lengths = overlap_max - overlap_min

        volumes = torch.prod(torch.clamp(overlap_lengths, min = 0), dim = -1)

        return volumes
    
    def get_gradient(self, joint_input, start, goal, t):

        joint_tensor1 = torch.tensor(joint_input, dtype = torch.float32, device = self.device)
        joint_tensor1.requires_grad = True
        joint_tensor1.grad = None

        # joint_tensor2 = torch.tensor(joint_input, dtype = torch.float32, device = self.device)
        # joint_tensor2.requires_grad = True
        # joint_tensor2.grad = None

        cost = torch.sum(self.cost(joint_tensor1, t))
        # smoothness_cost = self.smoothness_cost(joint_tensor2, start, goal)
        cost.backward()
        # smoothness_cost.backward()

        gradient1 = joint_tensor1.grad.cpu().numpy()
        # gradient2 = joint_tensor2.grad.cpu().numpy()

        # if np.linalg.norm(gradient2) > 0:    
        #     gradient2 = gradient2 / np.linalg.norm(gradient2)

        return gradient1 #+ 0.1 * gradient2
    
    def choose_best_trajectory(self, trajectories):

        joint_tensor = torch.tensor(trajectories, dtype = torch.float32, device = self.device)
        overlap_volumes = torch.sum(self.cost(joint_tensor, 0), dim = (1, 2))
        min_index = torch.argmin(overlap_volumes)

        # print(overlap_volumes)
        # print("best = ", overlap_volumes[min_index])

        return trajectories[min_index]
    
    def smoothness_cost(self, joints, start, goal):

        start = torch.tensor(start, dtype = torch.float32, device = self.device).view(1, 7, 1)
        goal = torch.tensor(goal, dtype = torch.float32, device = self.device).view(1, 7, 1)

        cost = torch.sum((joints[:, :, 0] - start)**2) + torch.sum((joints[:, :, 2:-1] - joints[:, :, 1:-2])**2) + torch.sum((goal - joints[:, :, -1])**2)

        return cost
    
    def smoothness_metric(self, joints, dt):
        """
        dt is time taken between each step
        joints is (7, 50) numpy array
        """


        joint_tensor = self.rearrange_joints(torch.tensor(joints, dtype = torch.float32, device = self.device).unsqueeze(0))
        end_eff_transforms = self.get_end_effector_transform(joint_tensor)

        end_eff_positions = (end_eff_transforms[0, :, :3, 3]).numpy(force=True)
        # end_eff_positions is (50, 3) numpy array

        reshaped_joints = joints.T
        # reshaped_joints is (50, 7) numpy array
        joint_smoothness = np.linalg.norm(np.diff(reshaped_joints, n=1, axis=0) / dt, axis=1)
        joint_smoothness = self.sparc(joint_smoothness, 1. / dt)
        
        end_eff_smoothness = np.linalg.norm(np.diff(end_eff_positions, n=1, axis=0) / dt, axis=1)
        end_eff_smoothness = self.sparc(end_eff_smoothness, 1. / dt)

        return joint_smoothness, end_eff_smoothness
    
    def path_length_metric(self, joints):

        joint_tensor = self.rearrange_joints(torch.tensor(joints, dtype = torch.float32, device = self.device).unsqueeze(0))
        end_eff_transforms = self.get_end_effector_transform(joint_tensor)
        end_eff_positions = (end_eff_transforms[0, :, :3, 3]).numpy(force=True)
        # end_eff_positions is (50, 3) numpy array

        reshaped_joints = joints.T

        end_eff_path_length = np.sum(np.linalg.norm(np.diff(end_eff_positions, 1, axis=0), axis=1))
        joint_path_length = np.sum(np.linalg.norm(np.diff(reshaped_joints, 1, axis=0), axis=1))

        return joint_path_length, end_eff_path_length

    def sparc(self, movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
        """
        Calculates the smoothness of the given speed profile using the modified
        spectral arc length metric.
        Parameters
        ----------
        movement : np.array
                The array containing the movement speed profile.
        fs       : float
                The sampling frequency of the data.
        padlevel : integer, optional
                Indicates the amount of zero padding to be done to the movement
                data for estimating the spectral arc length. [default = 4]
        fc       : float, optional
                The max. cut off frequency for calculating the spectral arc
                length metric. [default = 10.]
        amp_th   : float, optional
                The amplitude threshold to used for determing the cut off
                frequency upto which the spectral arc length is to be estimated.
                [default = 0.05]
        Returns
        -------
        sal      : float
                The spectral arc length estimate of the given movement's
                smoothness.
        (f, Mf)  : tuple of two np.arrays
                This is the frequency(f) and the magntiude spectrum(Mf) of the
                given movement data. This spectral is from 0. to fs/2.
        (f_sel, Mf_sel) : tuple of two np.arrays
                        This is the portion of the spectrum that is selected for
                        calculating the spectral arc length.
        Notes
        -----
        This is the modfieid spectral arc length metric, which has been tested only
        for discrete movements.

        Examples
        --------
        >>> t = np.arange(-1, 1, 0.01)
        >>> move = np.exp(-5*pow(t, 2))
        >>> sal, _, _ = sparc(move, fs=100.)
        >>> '%.5f' % sal
        '-1.41403'
        """
        if np.allclose(movement, 0):
            print("All movement was 0, returning 0")
            return 0, None, None
        # Number of zeros to be padded.
        nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

        # Frequency
        f = np.arange(0, fs, fs / nfft)
        # Normalized magnitude spectrum
        Mf = abs(np.fft.fft(movement, nfft))
        Mf = Mf / max(Mf)

        # Indices to choose only the spectrum within the given cut off frequency
        # Fc.
        # NOTE: This is a low pass filtering operation to get rid of high frequency
        # noise from affecting the next step (amplitude threshold based cut off for
        # arc length calculation).
        fc_inx = ((f <= fc) * 1).nonzero()
        f_sel = f[fc_inx]
        Mf_sel = Mf[fc_inx]

        # Choose the amplitude threshold based cut off frequency.
        # Index of the last point on the magnitude spectrum that is greater than
        # or equal to the amplitude threshold.
        inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
        fc_inx = range(inx[0], inx[-1] + 1)
        f_sel = f_sel[fc_inx]
        Mf_sel = Mf_sel[fc_inx]

        # Calculate arc length
        new_sal = -sum(
            np.sqrt(
                pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) + pow(np.diff(Mf_sel), 2)
            )
        )
        return new_sal, (f, Mf), (f_sel, Mf_sel)





