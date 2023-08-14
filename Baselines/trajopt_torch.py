import torch
import torch.nn as nn

import numpy as np

class TrajOpt(nn.Module):

    def __init__(self, env, num_joints, traj_len):

        super().__init__()
        
        self.env = env
        self.num_joints = num_joints
        self.traj_len = traj_len

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def initialize_problem(self, start_joints, goal_pose, initial_guess = None):

        if initial_guess == None:
            self.theta = torch.rand(self.traj_len, self.num_joints).to(self.device)
        else:
            self.theta = torch.tensor(initial_guess).to(self.device)
        self.theta.requires_grad = True

        self.start_joints = torch.tensor(start_joints).to(self.device)
        self.goal_transform = torch.tensor(self.env.pose_to_transformation(goal_pose)).to(self.device)
    
    def solve(self, epochs = 100):
        
        for _ in range(epochs):
            
            self.theta.grad = None
            
            cost = self.objective()
            cost.backward()

            with torch.no_grad():
                self.theta -= 0.002*self.theta.grad

        # while True:     # Constraint Penalty Loop

        #     exit_convexification = False

        #     while True:    # Convexification loop
                
        #         # Calculate initial cost for no change in theta for it:
        #         initial_cost = self.objective(0, mu, start_joints, goal_pose)
                
        #         while True:     # Trust Region Loop
                                        
        #             # Get the improved cost and the optimal value:
        #             new_cost = result.fun
        #             del_theta = result.x

        #             print("Distance Cost: ", self.length_objective(0), "\nConstraint Cost: ", self.get_constraint_cost(0))
        #             _ = input(" ")
                    
        #             # Calculate total true model improvement and current model improvement:
        #             model_improve = initial_cost - new_cost
        #             true_improve = prev_true_improve + model_improve

        #             # Expand or shrink trust region based on the fraction of current improvement over total improvement:
        #             if model_improve / true_improve > c:
        #                 s = s * trust_expansion_factor
        #                 break
        #             else:
        #                 s = s * trust_shrinkage_factor

        #             # End the convexification loop if the trust region size is smaller than the tolerance for theta:
        #             if np.all(s < xtol):
        #                 exit_convexification = True
        #                 break
                
        #         prev_true_improve = true_improve
                
        #         # If the improvement in theta and objective(theta) is less than a threshold break convexification loop:
        #         if model_improve < ftol and np.all(del_theta) < xtol:
        #             break
        #         if exit_convexification:
        #             break

        #         self.theta = self.theta + del_theta

        #     # End the constraint penalty loop if the constraints are satisfied by a certain tolerance:
        #     if np.all(self.get_constraint_cost(0) <= ctol):
        #         break
        #     else:
        #         mu = k * mu

    
    def objective(self):

        return self.length_cost(self.theta) + self.goal_cost(self.theta)
    
    def length_cost(self, theta):

        return torch.sum((theta[1:, :] - theta[:-1, :]) ** 2) + torch.sum((self.start_joints - theta[0, :]) ** 2)
    
    def get_constraint_cost(self, x, mu, start_joints, goal_pose):

        new_joints = self.theta + x

        start_constraint = np.sum((new_joints[:self.num_joints] - start_joints) ** 2)
        goal_constraint = np.sum((self.env.forward_kinematics(new_joints[-self.num_joints:]) - self.env.pose_to_transformation(goal_pose)) ** 2)

        constraint_cost = mu * (start_constraint + goal_constraint)

        return constraint_cost
    
    def lower_joint_inequality_constraint(self, x):

        return (self.theta + x) - np.repeat([self.env.joint_lower_limits], self.traj_len, axis = 0).flatten()
    
    def upper_joint_inequality_constraint(self, x):

        return np.repeat([self.env.joint_upper_limits], self.traj_len, axis = 0).flatten() - (self.theta + x)
    
    def goal_cost(self, theta):

        return torch.sum((self.forward_kinematics(theta[-1, :]).to(self.device) - self.goal_transform)**2)
    
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

    def forward_kinematics(self, joint_angles):

        dh_params = torch.tensor([[0, 0.333, 0, 0],
                                  [0, 0, -torch.pi / 2, 0],
                                  [0, 0.316, torch.pi / 2, 0],
                                  [0.0825, 0, torch.pi / 2, 0],
                                  [-0.0825, 0.384, -torch.pi / 2, 0],
                                  [0, 0, torch.pi / 2, 0],
                                  [0.088, 0, torch.pi / 2, 0],
                                  [0, 0.107, 0, 0],
                                  [0, 0, 0, -torch.pi / 4],
                                  [0.0, 0.1034, 0, 0]])
        
        dh_params[:7, 3] = joint_angles
        
        T_EE = torch.eye(4)
        for i in range(7 + 3):
            T_EE = T_EE @ self.get_tf_mat(i, dh_params)

        return T_EE
    