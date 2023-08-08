import numpy as np
import pybullet as p
import math
import pytorch_kinematics as pk
import pybullet_data
import pybullet_utils.bullet_client as bc


class Differentiable_Panda:
    def __init__(self) -> None:
        self.panda_chain = pk.build_serial_chain_from_urdf(open("/home/vishal/anaconda3/envs/mpinets2/lib/python3.8/site-packages/pybullet_data/franka_panda/panda.urdf").read(), "panda_hand")
        print(self.panda_chain)

        print(self.panda_chain.get_joint_parameter_names())

        q = np.array([-0.5778415 ,  0.54352427,  0.6913696 , -2.5508945 , -0.11496189, 3.0910137, -2.49674, 0, 0  ])

        print(f"Forward kinematics: {self.panda_chain.forward_kinematics(q, end_only=False)}")

    def get_forward_kinematics(self, q):
        poses = self.panda_chain.forward_kinematics(q)

        return poses
    
if __name__=="__main__":
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

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    dPanda = Differentiable_Panda()
