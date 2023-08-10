
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
from diffusion.infer_diffusion import infer, infer_guided, value_func_and_grad
from train_SDF import SDFModel

lower_lims = np.array([-2.8973000049591064, -1.7627999782562256, -2.8973000049591064, -3.0717999935150146, -2.8973000049591064, -0.017500000074505806, -2.8973000049591064])
upper_lims = np.array([2.8973000049591064, 1.7627999782562256, 2.8973000049591064, -0.0697999969124794, 2.8973000049591064, 3.752500057220459, 2.8973000049591064])

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

def generate_collision_course4(client_id):
    obstacle_centers = []
    sphereRadius = 0.08
    colVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
    # colSphereId = client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    obstacle_center = np.array([0.6, -0.10, 0.7])
    obstacle_centers.append(obstacle_center)
    colSph = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

    colVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
    # colSphereId = client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    obstacle_center = np.array([0.58, -0.09, 0.3])
    obstacle_centers.append(obstacle_center)
    colSph = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

    colVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
    # colSphereId = client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    obstacle_center = np.array([0.4, -0.16, 0.4])
    obstacle_centers.append(obstacle_center)
    colSph = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

    q0 = np.array([-0.0493874, 1.2605693, 0.36301115, -0.26907736, -0.2328229, 1.1613771, -1.6480199 ]) 
    qTarget = np.array([-1.5038756, 0.9193187, 1.8278371, -1.4488376, -1.1391141, 1.2231112, -0.9670128])

    st_pos = np.array([0.75024617,  0.01952344,  0.32572699])
    end_pos = np.array([0.53903937, -0.27098262,  0.58423716])

    return obstacle_centers, q0, qTarget, st_pos, end_pos

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

obstacle_centers, q0, qTarget, st_pos, end_pos = generate_collision_course4(client_id=client_id)

sphereRadius = 0.05
inVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 1, 1, 1])
inViz = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=inVizshape, basePosition=st_pos) # np.array([0.4372005 , -0.14387664,  0.5400902]))

sphereRadius = 0.05
outVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 0, 1, 1])
outViz = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=outVizshape, basePosition=end_pos) #  np.array([0.2937204 ,  0.10894514,  0.5569069]))

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

print(getMotorJointStates(panda))


print("Robot info ended!!!!")

# sleep(10)

st_time = time.time()
# trajectory = infer(denoiser, q0, qTarget)
# trajectory = infer_guided(denoiser, q0, qTarget, obstacle_centers, client_id, panda, panda_joints)  #(50, 7)
trajectory = np.zeros((50, 7))
end_time = time.time()
print("-"*10)
print(f"Total Inference Time: {end_time - st_time} seconds")
print("-"*10)
print('traj shape: {}'.format(trajectory.shape))

# for l, joint_ind in enumerate(panda_joints):
#     client_id.resetJointState(panda, joint_ind, q0[l])

num_samples = 50000  # You can change this to generate a different number of samples
trajectory = np.random.uniform(lower_lims, upper_lims, size=(num_samples, len(lower_lims)))  #(50000, 7)
print('sdf joint angle dataset shape: {}'.format(trajectory.shape))
file_name = "joint_states.npy"
np.save(file_name, trajectory)

dists = np.zeros((trajectory.shape[0], trajectory.shape[1], len(obstacle_centers)), dtype=float)  #3D matrix , verify in pybullet
for conf_no in range(len(trajectory)):  #50 for mpinets
    for l, joint_ind in enumerate(panda_joints):
        client_id.resetJointState(panda, joint_ind, trajectory[conf_no, l])
    numJoints = p.getNumJoints(panda)  # 12 but consider only fisrt 7 joints
    for joint_no in range(7):# numJoints):
        link_pos = client_id.getLinkState(panda, joint_no, computeForwardKinematics=1)[0]
        print(link_pos)
        grad = None
        for obstacle_no in range(len(obstacle_centers)):
            value, grad = value_func_and_grad(obstacle_centers[obstacle_no], np.array(link_pos))
            # print(value)
            dists[conf_no, joint_no, obstacle_no] = value
    
    print('**************')

print('dits shape: {}'.format(dists.shape))
file_name = "sdf_dists_three_spheres.npy"

# Save the 3D numpy array to the specified file
np.save(file_name, dists)

# train sdf on distance to obstacles data


# sleep(5)
# for i in range(len(trajectory)):
#     sleep(0.4)
#     current_joints = np.array([client_id.getJointState(panda, i)[0] for i in panda_joints])
#     target_joints = trajectory[i]
#     print(f"Current Joints: {current_joints}")
#     print(f"Target Joints: {target_joints}")
#     print(f"Itr number: {i}")
#     move_joints(client_id, panda, panda_joints, target_joints)
#     client_id.stepSimulation()

# sleep(100)