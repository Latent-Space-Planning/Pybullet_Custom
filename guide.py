import numpy as np
import torch

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
                                  [0.0, 0.1034, 0, 0]])
    
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

