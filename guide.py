import numpy as np
import torch
import os
import re

class IntersectionVolumeGuide:

    def __init__(self, env):

        self.env = env
        self.static_dh_params = torch.tensor([[0, 0.333, 0, 0],
                                  [0, 0, -torch.pi / 2, 0],
                                  [0, 0.316, torch.pi / 2, 0],
                                  [0.0825, 0, torch.pi / 2, 0],
                                  [-0.0825, 0.384, -torch.pi / 2, 0],
                                  [0, 0, torch.pi / 2, 0],
                                  [0.088, 0, torch.pi / 2, 0],
                                  [0, 0.107, 0, 0],
                                  [0, 0, 0, -torch.pi / 4],
                                  [0.0, 0.1034, 0, 0]], dtype = torch.float32)
        
        self.define_link_information()
    
    def get_tf_mat(self, i, dh_params):
        
        a = dh_params[i, 0]
        d = dh_params[i, 1]
        alpha = dh_params[i, 2]
        q = dh_params[i, 3]

        transform = torch.eye(4)

        transform[0, 0] = torch.cos(q)
        transform[0, 1] = -torch.sin(q)
        transform[0, 3] = a

        transform[1, 0] = torch.sin(q) * torch.cos(alpha)
        transform[1, 1] = torch.cos(q) * torch.cos(alpha)
        transform[1, 2] = -torch.sin(alpha)
        transform[1, 3] = -torch.sin(alpha) * d

        transform[2, 0] = torch.sin(q) * torch.sin(alpha)
        transform[2, 1] = torch.cos(q) * torch.sin(alpha)
        transform[2, 2] = torch.cos(alpha)
        transform[2, 3] = torch.cos(alpha) * d

        return transform
    
    def forward_kinematics(self, joint_index, joint_angles):
        """
        joint_index => 1 to 10 (But we'll only use 1 to 7)
        joint_angles => Array of size 7
        """

        dh_params = torch.clone(self.static_dh_params)
        dh_params[:7, 3] = joint_angles
    
        T = torch.eye(4)
        for i in range(joint_index):
            T = T @ self.get_tf_mat(i, dh_params)

        return T
    
    def get_vertices(self, link_dimensions):

        l, b, h = link_dimensions[0], link_dimensions[1], link_dimensions[2]
        
        return torch.tensor([[-l/2, -b/2, -h/2, 1.],
                             [ l/2, -b/2, -h/2, 1.],
                             [ l/2,  b/2, -h/2, 1.],
                             [-l/2,  b/2, -h/2, 1.],
                             [-l/2, -b/2,  h/2, 1.],
                             [ l/2, -b/2,  h/2, 1.],
                             [ l/2,  b/2,  h/2, 1.],
                             [-l/2,  b/2,  h/2, 1.]], dtype = torch.float32)
    
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
        self.link_vertices = []
        
        for link_index in range(len(self.link_index_to_name)):

            link_name = self.link_index_to_name[link_index]
            link_dimensions = self.link_dimensions_from_name[link_name].copy()
            
            if link_index == len(self.link_index_to_name) - 1:
                link_dimensions[1] *= 4
            
            self.link_dimensions.append(link_dimensions)
            self.link_vertices.append(self.get_vertices(link_dimensions))

        self.link_static_joint_frame = [1, 2, 3, 4, 5, 6, 7, 7, 7] 
        self.static_frames = []

        # Link 0:
        self.static_frames.append(torch.tensor([[1., 0., 0., 8.71e-05],
                                                [0., 1., 0., -3.709035e-02],
                                                [0., 0., 1., -6.851545e-02],
                                                [0., 0., 0., 1.]], dtype = torch.float32))
        # Link 1:
        self.static_frames.append(torch.tensor([[1., 0., 0., -8.425e-05],
                                                [0., 1., 0., -6.93950016e-02],
                                                [0., 0., 1., 3.71961970e-02],
                                                [0., 0., 0., 1.]], dtype = torch.float32))
        
        # Link 2:
        self.static_frames.append(torch.tensor([[1., 0., 0., 0.0414576],
                                                [0., 1., 0., 0.0281429],
                                                [0., 0., 1., -0.03293086],
                                                [0., 0., 0., 1.]], dtype = torch.float32))
        
        # Link 3:
        self.static_frames.append(torch.tensor([[1., 0., 0., -4.12337575e-02],
                                                [0., 1., 0., 3.44296512e-02],
                                                [0., 0., 1., 2.79226985e-02],
                                                [0., 0., 0., 1.]], dtype = torch.float32))
        
        # Link 4:
        self.static_frames.append(torch.tensor([[1., 0., 0., 3.3450000e-05],
                                                [0., 1., 0., 3.7388050e-02],
                                                [0., 0., 1., -1.0619285e-01],
                                                [0., 0., 0., 1.]], dtype = torch.float32))
        
        # Link 5:
        self.static_frames.append(torch.tensor([[ 1., 0., 0., 4.21935000e-02],
                                                [ 0., 1., 0., 1.52195003e-02],
                                                [ 0., 0., 1., 6.07699933e-03],
                                                [ 0., 0., 0., 1.]], dtype = torch.float32))
        
        # Link 6:
        self.static_frames.append(torch.tensor([[ 1., 0., 0., 1.86357500e-02],
                                                [ 0., 1., 0., 1.85788569e-02],
                                                [ 0., 0., 1., 7.94137484e-02],
                                                [ 0., 0., 0., 1.]], dtype = torch.float32))
        
        # Link 7:
        self.static_frames.append(torch.tensor([[ 7.07106767e-01, 7.07106795e-01, 0., -1.26717073e-03],
                                                [-7.07106795e-01, 7.07106767e-01, 0., -1.25294673e-03],
                                                [ 0., 0., 1., 1.27018693e-01],
                                                [ 0., 0., 0., 1.]], dtype = torch.float32))
        
        # Link 8:
        self.static_frames.append(torch.tensor([[ 7.07106767e-01, 7.07106795e-01, 0., 9.29352476e-03],
                                                [-7.07106795e-01, 7.07106767e-01, 0., 9.28272434e-03],
                                                [ 0., 0., 1., 1.92390375e-01],
                                                [ 0., 0., 0., 1.]], dtype = torch.float32))
        
    def get_link_transform(self, link_index, joint_angles):

        joint_transform = self.forward_kinematics(self.link_static_joint_frame[link_index], joint_angles)

        link_transform = joint_transform @ self.static_frames[link_index]

        return link_transform
    
    def volume_cost(self, joint_angles, obstacle_pose, obstacle_size):

        obstacle_static_vertices = self.get_vertices(obstacle_size)
        obstacle_rot = np.array(self.env.client_id.getMatrixFromQuaternion(obstacle_pose[3:])).reshape((3, 3))

        obstacle_transform = np.eye(4)
        obstacle_transform[:3, :3] = obstacle_rot.copy()
        obstacle_transform[:3, -1] = obstacle_pose[:3]

        obstacle_transform = torch.tensor(obstacle_transform, dtype = torch.float32)

        obstacle_vertices = obstacle_transform @ (obstacle_static_vertices.T)
        obs_min = torch.min(obstacle_vertices, dim = 1)[0]
        obs_max = torch.max(obstacle_vertices, dim = 1)[0]

        # print("Obstacle Transform = ", obstacle_transform)
        # print("Obstacle Vertices = ", obstacle_vertices)
        # print("Obstacle Min = ", obs_min)
        # print("Obstacle Max = ", obs_max)

        cost = 0

        # print("-------------------------------------------")

        for link_index in range(9):

            link_transform = self.get_link_transform(link_index, joint_angles)
            link_vertices = link_transform @ (self.link_vertices[link_index].T)

            link_min = torch.min(link_vertices, dim = 1)[0]
            link_max = torch.max(link_vertices, dim = 1)[0]

            overlap_min = torch.max(link_min, obs_min)[:3]
            overlap_max = torch.min(link_max, obs_max)[:3]

            overlap_lengths = overlap_max - overlap_min
            volume = torch.prod(torch.clamp(overlap_lengths, min = 0))
            
            # print("Link_index = ", link_index)
            # print("Link Transform = ", link_transform)
            # print("Link Vertices = ", link_vertices)

            # print("Link Min = ", link_min)
            # print("Link Max = ", link_max)

            # print("Overlap Min = ", overlap_min)
            # print("Overlap Max = ", overlap_max)
            # print("Overlap Lengths = ", overlap_lengths)
            # print("Volume = ", volume)

            # print("--------------------------------------------")

            cost = cost + volume

        return cost




