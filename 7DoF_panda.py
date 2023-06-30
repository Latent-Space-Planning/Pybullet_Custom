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
sys.path.append('/home/vishal/Volume_E/Active/Undergrad_research/CoRL2023/Latent_Space_Organization/Mpinet_Environment/')


gui =True
timestep = 1/480


client_id = bc.BulletClient(p.GUI if gui else p.DIRECT)
client_id.setAdditionalSearchPath(pybullet_data.getDataPath())
client_id.setTimeStep(timestep)
client_id.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())



target = client_id.getDebugVisualizerCamera()[11]
client_id.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=90,
    cameraPitch=-25,
    cameraTargetPosition=target,
)


p.resetSimulation()
client_id.setGravity(0, 0, -9.8)

plane = client_id.loadURDF("plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True)
workspace = client_id.loadURDF(
    "./Assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True
)
client_id.changeDynamics(
    plane,
    -1,
    lateralFriction=1.1,
    restitution=0.5,
    linearDamping=0.5,
    angularDamping=0.5,
)
client_id.changeDynamics(
    workspace,
    -1,
    lateralFriction=1.1,
    restitution=0.5,
    linearDamping=0.5,
    angularDamping=0.5,
)

panda = client_id.loadURDF("franka_panda/panda.urdf", basePosition=(0, 0, 0), useFixedBase=True)
panda_joints = []
for i in range(client_id.getNumJoints(panda)):
    info = client_id.getJointInfo(panda, i)
    joint_id = info[0]
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    # if joint_name == "ee_fixed_joint":
        # panda_ee_id = joint_id
    if joint_type == client_id.JOINT_REVOLUTE:
        panda_joints.append(joint_id)
# client_id.enableJointForceTorqueSensor(panda, panda_ee_id, 1)



def move_joints(client_id, panda, panda_joints, target_joints, speed=0.01, timeout=3):
    """Move UR5e to target joint configuration."""
    t0 = time.time()
    all_joints = []
    while (time.time() - t0) < timeout:
        current_joints = np.array([client_id.getJointState(panda, i)[0] for i in panda_joints])
        # pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
        # if pos[2] < 0.005:
        #     print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
        #     return False, []
        diff_joints = target_joints - current_joints
        if all(np.abs(diff_joints) < 1e-2):
            # give time to stop
            for _ in range(10):
                client_id.stepSimulation()
            return True, all_joints

        # Move with constant velocity
        norm = np.linalg.norm(diff_joints)
        v = diff_joints / norm if norm > 0 else 0
        step_joints = current_joints + v * speed
        all_joints.append(step_joints)
        client_id.setJointMotorControlArray(
            bodyIndex=panda,
            jointIndices=panda_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=step_joints,
            positionGains=np.ones(len(panda_joints)),
        )
        client_id.stepSimulation()
    print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
    return False, []


# Creating collision Sphere
sphereRadius = 0.08

colVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
# colSphereId = client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
obstacle_center = np.array([0.42, 0.25, 0.4])

colSph = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)



sphereRadius = 0.05
inVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 1, 1, 1])
inViz = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=inVizshape, basePosition=np.array([0.44372806, 0.09607968 ,0.1356831])) # np.array([0.4372005 , -0.14387664,  0.5400902]))

sphereRadius = 0.05
outVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 0, 1, 1])
outViz = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=outVizshape, basePosition=np.array([0.39664501,  0.4099783,   0.16045232])) # np.array([0.2937204 ,  0.10894514,  0.5569069]))


# Inferring from Diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
traj_len=50
T = 255
model_name = "./diffusion/checkpoints/7dof/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)
if not os.path.exists(model_name):
    print("Model does not exist for these parameters. Train a model first.")
    _ = input("Press anything to exit")
    exit()
denoiser = TemporalUNet(model_name = model_name, input_dim = 7, time_dim = 32, dims=(32, 64, 128, 256, 512, 512))
_ = denoiser.to(device)

q0 = np.array([-0.5778415 ,  0.54352427,  0.6913696 , -2.5508945 , -0.11496189, 3.0910137, -2.49674   ]) # np.array([ 0.9798614,  0.5573511, -0.1629493, -1.9912002, -0.23168434, 2.214461, -0.38091224])
qTarget = np.array([ 0.9798614,  0.5573511, -0.1629493, -1.9912002, -0.23168434, 2.214461, -0.38091224])  # np.array([2.2053013 ,  1.2142502 , -2.5370142 , -2.4402852 ,  1.0121354 ,1.2068753 , -1.7239444]) # 


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

print(getMotorJointStates(panda))


print("Robot info ended!!!!")

# sleep(10)

trajectory = infer(denoiser, q0, qTarget)
# trajectory = infer_guided(denoiser, q0, qTarget, obstacle_center, client_id, panda, panda_joints)

for i, joint_ind in enumerate(panda_joints):
    client_id.resetJointState(panda, joint_ind, q0[i])

for i in range(len(trajectory)):
    sleep(0.4)
    current_joints = np.array([client_id.getJointState(panda, i)[0] for i in panda_joints])
    target_joints = trajectory[i]
    print(f"Current Joints: {current_joints}")
    print(f"Target Joints: {target_joints}")
    print(f"Itr number: {i}")
    move_joints(client_id, panda, panda_joints, target_joints)
    client_id.stepSimulation()

sleep(100)