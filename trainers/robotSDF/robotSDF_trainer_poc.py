import wandb
import pybullet as p
import numpy as np
import argparse
import random
import torch

from Environment.panda_7DoF import PandaEnv

if __name__ == "__main__":
    env = PandaEnv()
    