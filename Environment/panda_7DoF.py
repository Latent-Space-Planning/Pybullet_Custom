import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
from time import sleep
import sys
import time
import numpy as np
import torch
import os
from diffusion.Models.temporalunet import TemporalUNet
from diffusion.infer_diffusion import infer, infer_guided
# sys.path.append('/home/vishal/Volume_E/Active/Undergrad_research/CoRL2023/Latent_Space_Organization/Mpinet_Environment/')


class PandaEnv:
    def __init__(self) -> None:
        
        self.gui =True
        self.timestep = 1/480


        self.client_id = bc.BulletClient(p.GUI if self.gui else p.DIRECT)
        self.client_id.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client_id.setTimeStep(self.timestep)
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())



        target = self.client_id.getDebugVisualizerCamera()[11]
        self.client_id.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-25,
            cameraTargetPosition=target,
        )


        p.resetSimulation()
        self.client_id.setGravity(0, 0, -9.8)

        plane = self.client_id.loadURDF("plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True)
        workspace = self.client_id.loadURDF(
            "./Assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True
        )
        self.client_id.changeDynamics(
            plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        self.client_id.changeDynamics(
            workspace,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        self.panda = self.client_id.loadURDF("franka_panda/panda.urdf", basePosition=(0, 0, 0), useFixedBase=True)
        self.panda_joints = []
        for i in range(self.client_id.getNumJoints(self.panda)):
            info = self.client_id.getJointInfo(self.panda, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            # if joint_name == "ee_fixed_joint":
                # panda_ee_id = joint_id
            if joint_type == self.client_id.JOINT_REVOLUTE:
                self.panda_joints.append(joint_id)
        # self.client_id.enableJointForceTorqueSensor(panda, panda_ee_id, 1)



    def move_joints(self, panda, panda_joints, target_joints, speed=0.01, timeout=3):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        all_joints = []
        while (time.time() - t0) < timeout:
            current_joints = np.array([self.client_id.getJointState(panda, i)[0] for i in panda_joints])
            # pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
            # if pos[2] < 0.005:
            #     print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
            #     return False, []
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 1e-2):
                # give time to stop
                for _ in range(10):
                    self.client_id.stepSimulation()
                return True, all_joints

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            all_joints.append(step_joints)
            self.client_id.setJointMotorControlArray(
                bodyIndex=panda,
                jointIndices=panda_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(panda_joints)),
            )
            self.client_id.stepSimulation()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False, []


    # Creating collision Sphere
    def generate_collision_course(self):
        obstacle_centers = []
        sphereRadius = 0.08
        colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        obstacle_center = np.array([0.46, 0.30, 0.3])
        obstacle_centers.append(obstacle_center)
        colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        obstacle_center = np.array([0.36, 0.19, 0.4])
        obstacle_centers.append(obstacle_center)
        colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        obstacle_center = np.array([0.19, 0.51, 0.3])
        obstacle_centers.append(obstacle_center)
        colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        q0 = np.array([-0.5778415 ,  0.54352427,  0.6913696 , -2.5508945 , -0.11496189, 3.0910137, -2.49674   ])
        qTarget = np.array([ 0.9798614,  0.5573511, -0.1629493, -1.9912002, -0.23168434, 2.214461, -0.38091224]) 

        st_pos = np.array([0.44372806, 0.09607968 ,0.1356831])
        end_pos = np.array([0.39664501,  0.4099783,   0.16045232])

        return obstacle_centers, q0, qTarget, st_pos, end_pos

    def generate_collision_course2(self):
        obstacle_centers = []

        for i in range(14):
            sphereRadius = 0.05
            colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
            # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
            obstacle_center = np.array([0.45, 0.25, 0.3+i*0.05])
            obstacle_centers.append(obstacle_center)
            colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        # colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        # obstacle_center = np.array([0.42, 0.25, 0.38])
        # obstacle_centers.append(obstacle_center)
        # colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        # colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        # obstacle_center = np.array([0.42, 0.25, 0.54])
        # obstacle_centers.append(obstacle_center)
        # colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)
        q0 = np.array([-0.5778415 ,  0.54352427,  0.6913696 , -2.5508945 , -0.11496189, 3.0910137, -2.49674   ])
        qTarget = np.array([ 0.9798614,  0.5573511, -0.1629493, -1.9912002, -0.23168434, 2.214461, -0.38091224]) 

        st_pos = np.array([0.44372806, 0.09607968 ,0.1356831])
        end_pos = np.array([0.39664501,  0.4099783,   0.16045232])

        return obstacle_centers, q0, qTarget, st_pos, end_pos

    def generate_collision_course3(self):
        obstacle_centers = []
        sphereRadius = 0.08
        colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        obstacle_center = np.array([0.6, -0.10, 0.6])
        obstacle_centers.append(obstacle_center)
        colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        obstacle_center = np.array([0.55, -0.09, 0.3])
        obstacle_centers.append(obstacle_center)
        colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        obstacle_center = np.array([0.4, -0.1, 0.4])
        obstacle_centers.append(obstacle_center)
        colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        q0 = np.array([-0.0493874, 1.2605693, 0.36301115, -0.26907736, -0.2328229, 1.1613771, -1.6480199 ]) 
        qTarget = np.array([-1.5038756, 0.9193187, 1.8278371, -1.4488376, -1.1391141, 1.2231112, -0.9670128])

        st_pos = np.array([0.75024617,  0.01952344,  0.32572699])
        end_pos = np.array([0.53903937, -0.27098262,  0.58423716])

        return obstacle_centers, q0, qTarget, st_pos, end_pos

    def generate_collision_course4(self):
        obstacle_centers = []
        sphereRadius = 0.08
        colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        obstacle_center = np.array([0.6, -0.10, 0.7])
        obstacle_centers.append(obstacle_center)
        colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        obstacle_center = np.array([0.58, -0.09, 0.3])
        obstacle_centers.append(obstacle_center)
        colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        colVizshape = self.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
        # colSphereId = self.client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        obstacle_center = np.array([0.4, -0.16, 0.4])
        obstacle_centers.append(obstacle_center)
        colSph = self.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

        q0 = np.array([-0.0493874, 1.2605693, 0.36301115, -0.26907736, -0.2328229, 1.1613771, -1.6480199 ]) 
        qTarget = np.array([-1.5038756, 0.9193187, 1.8278371, -1.4488376, -1.1391141, 1.2231112, -0.9670128])

        st_pos = np.array([0.75024617,  0.01952344,  0.32572699])
        end_pos = np.array([0.53903937, -0.27098262,  0.58423716])

        return obstacle_centers, q0, qTarget, st_pos, end_pos
    

    def getMotorJointStates(self, robot):
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques
    
    def resetPandaJoints(self, q):
        for i, joint_ind in enumerate(self.panda_joints):
            self.client_id.resetJointState(self.panda, joint_ind, q[i])

    def checkPandaJointLimits(self, qbatch):
        '''
        Check if the given batch of panda configurations fall within the joint limits
        '''
        raise NotImplementedError("Not implemented yet!")
    
    def getClosestDistanceToScene(self, distThresh=0.45):
        '''
        Gets the smallest distance between the robot and the scene (collision objects). If the distance > distThresh, 
        it returns distThresh


        Parameters:
        1. distThresh: If closest dist < distThresh, we return closest distance, otherwise, we return distThresh
        '''
        


    def samplePandaConfig(self, batch_size):
        '''
        Generate a batch of valid configurations which may or may not collide with the environment
        '''
        q = np.random.uniform(low=self.panda_lower_lims, high=self.panda_higher_lims, size=(batch_size, 7))
        return q




if __name__=="__main__":
    env = PandaEnv()

    obstacle_centers, q0, qTarget, st_pos, end_pos = env.generate_collision_course4()

    sphereRadius = 0.05
    inVizshape = env.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 1, 1, 1])
    inViz = env.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=inVizshape, basePosition=st_pos) # np.array([0.4372005 , -0.14387664,  0.5400902]))

    sphereRadius = 0.05
    outVizshape = env.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 0, 1, 1])
    outViz = env.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=outVizshape, basePosition=end_pos) #  np.array([0.2937204 ,  0.10894514,  0.5569069]))


    # Inferring from Diffusion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    traj_len=50
    T = 255
    model_name = "/home/jayaram/research/research_tracks/table_top_rearragement/global_classifier_guidance_for_7DOF_manipulator/diffuser_ckpts_7dof_mpinets/7dof/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)
    if not os.path.exists(model_name):
        print("Model does not exist for these parameters. Train a model first.")
        _ = input("Press anything to exit")
        exit()
    denoiser = TemporalUNet(model_name = model_name, input_dim = 7, time_dim = 32, dims=(32, 64, 128, 256, 512, 512))
    _ = denoiser.to(device)

    # q0 = np.array([ 0.9798614,  0.5573511, -0.1629493, -1.9912002, -0.23168434, 2.214461, -0.38091224])
    # qTarget = np.array([2.2053013 ,  1.2142502 , -2.5370142 , -2.4402852 ,  1.0121354 ,1.2068753 , -1.7239444]) # 


    # getLinkStates(panda, )

    print("Robot Information!!!")

    def getMotorJointStates(robot):
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    print(getMotorJointStates(env.panda))


    print("Robot info ended!!!!")

    # sleep(10)

    st_time = time.time()
    # trajectory = infer(denoiser, q0, qTarget)
    trajectory = infer_guided(denoiser, q0, qTarget, obstacle_centers, env.client_id, env.panda, env.panda_joints)
    end_time = time.time()
    print("-"*10)
    print(f"Total Inference Time: {end_time - st_time} seconds")
    print("-"*10)

    for i, joint_ind in enumerate(env.panda_joints):
        env.client_id.resetJointState(env.panda, joint_ind, q0[i])

    sleep(5)
    for i in range(len(trajectory)):
        sleep(0.4)
        current_joints = np.array([env.client_id.getJointState(env.panda, i)[0] for i in env.panda_joints])
        target_joints = trajectory[i]
        print(f"Current Joints: {current_joints}")
        print(f"Target Joints: {target_joints}")
        print(f"Itr number: {i}")
        env.move_joints(env.panda, env.panda_joints, target_joints)
        env.client_id.stepSimulation()

    sleep(100)