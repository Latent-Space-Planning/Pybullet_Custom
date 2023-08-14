import scipy as sp
from scipy.optimize import minimize
import numpy as np

class TrajOpt:

    def __init__(self, env, num_joints, traj_len):

        self.env = env
        self.num_joints = num_joints
        self.traj_len = traj_len

    def solve(self, start_joints, goal_pose, 
              c = 0.1, 
              mu = 10000,
              k = 2,
              trust_expansion_factor = 1.5,
              trust_shrinkage_factor = 0.8,
              xtol = 1e-4,
              ftol = 1e-3, 
              ctol = 1e-5, 
              initial_guess = None):
        
        # Set the initial guess for joint angles theta:
        if initial_guess == None:
            self.theta = np.random.rand(self.traj_len * self.num_joints)
        else:
            self.theta = np.reshape(np.array(initial_guess), (self.traj_len * self.num_joints,)).copy()
        self.theta[:self.num_joints] = np.array(start_joints).copy()

        # Define trust region bounds:
        s = np.array([(self.env.joint_upper_limits - self.env.joint_lower_limits) / 1e+3])
        s = np.repeat(s, self.traj_len, axis = 0).flatten()
        
        # Define joint limit constraints:
        joint_limit_constraints = [{'type': 'ineq', 'fun': self.lower_joint_inequality_constraint},
                                   {'type': 'ineq', 'fun': self.upper_joint_inequality_constraint},
                                   {'type': 'eq', 'fun': self.start_constraint, 'args': (start_joints,)},
                                   {'type': 'eq', 'fun': self.goal_constraint, 'args': (goal_pose,)}]

        # Initialize the true improve as zero
        prev_true_improve = 0

        while True:     # Constraint Penalty Loop

            exit_convexification = False

            while True:    # Convexification loop
                
                # Calculate initial cost for no change in theta for it:
                initial_cost = self.objective(0, mu, start_joints, goal_pose)
                
                while True:     # Trust Region Loop
                    
                    bounds = tuple([(-s[i], s[i]) for i in range(s.size)])
                    
                    print(bounds)
                    print(s)
                    print((mu, start_joints, goal_pose))
                    _ = input(" ")
                    print("Started Optimizing")

                    # Perform the optimization of the objective within the trust region:
                    result = minimize(fun = self.objective,
                                      x0 = np.zeros_like(self.theta),
                                      args = (mu, start_joints, goal_pose),
                                      method = 'SLSQP',
                                      bounds = bounds,
                                      constraints = joint_limit_constraints)
                    
                    # Get the improved cost and the optimal value:
                    new_cost = result.fun
                    del_theta = result.x

                    print("Distance Cost: ", self.length_objective(0), "\nConstraint Cost: ", self.get_constraint_cost(0))
                    _ = input(" ")
                    
                    # Calculate total true model improvement and current model improvement:
                    model_improve = initial_cost - new_cost
                    true_improve = prev_true_improve + model_improve

                    # Expand or shrink trust region based on the fraction of current improvement over total improvement:
                    if model_improve / true_improve > c:
                        s = s * trust_expansion_factor
                        break
                    else:
                        s = s * trust_shrinkage_factor

                    # End the convexification loop if the trust region size is smaller than the tolerance for theta:
                    if np.all(s < xtol):
                        exit_convexification = True
                        break
                
                prev_true_improve = true_improve
                
                # If the improvement in theta and objective(theta) is less than a threshold break convexification loop:
                if model_improve < ftol and np.all(del_theta) < xtol:
                    break
                if exit_convexification:
                    break

                self.theta = self.theta + del_theta

            # End the constraint penalty loop if the constraints are satisfied by a certain tolerance:
            if np.all(self.get_constraint_cost(0) <= ctol):
                break
            else:
                mu = k * mu

    
    def objective(self, x, mu, start_joints, goal_pose):

        return self.length_objective(x) #+ self.get_constraint_cost(x, mu, start_joints, goal_pose)
    
    def length_objective(self, x):

        new_joints = self.theta + x        

        return np.sum((new_joints[self.num_joints:] - new_joints[:-self.num_joints])**2)
    
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
    
    def start_constraint(self, x, start_joints):

        new_joints = self.theta + x
        return new_joints[:self.num_joints] - start_joints
    
    def goal_constraint(self, x, goal_pose):

        new_joints = self.theta + x
        return self.env.forward_kinematics(new_joints[-self.num_joints:]) - self.env.pose_to_transformation(goal_pose)