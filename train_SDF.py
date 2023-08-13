import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import time
from training import Trainer_jayaram
from torch.utils.data import Dataset
import os
import numpy as np


import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import wandb

# wandb setup
number = 1
NAME = "model" + str(number)
ID = 'sdf_training single scene' + str(number)
# run = wandb.init(project='sdf_training', name = NAME, id = ID)

# Define the Signed Distance Function MLP model
class SDFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):
        super(SDFModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)   # Fully connected layer with 7 input features and 32 output features
        self.relu = nn.ReLU()         # ReLU activation function
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)  # Fully connected layer with 32 input features and 128 output features
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)  # Fully connected layer with 128 input features and 64 output features
        self.fc4 = nn.Linear(hidden_dim_3, output_dim)  # Fully connected layer with 64 input features and 21 output features

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        print(x.requires_grad)
        x = torch.view(-1, 7, 3)   # comment this during inference
        return x
    
    # def __init__(self, input_dim, hidden_dim, output_dim):
    #     super(SDFModel, self).__init__()
    #     self.layers = nn.Sequential(
    #         nn.Linear(input_dim, hidden_dim),
    #         nn.ReLU(),
    #         nn.Linear(hidden_dim, hidden_dim),
    #         nn.ReLU(),
    #         nn.Linear(hidden_dim, output_dim)
    #     )

    # def forward(self, x):
    #     return self.layers(x)

# Define the Signed Distance Function MLP model
class SDFModel_min_dist_per_joint(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):
        super(SDFModel_min_dist_per_joint, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)   # Fully connected layer with 7 input features and 32 output features
        self.relu = nn.ReLU()         # ReLU activation function
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)  # Fully connected layer with 32 input features and 128 output features
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)  # Fully connected layer with 128 input features and 32 output features
        self.fc4 = nn.Linear(hidden_dim_3, output_dim)    # Fully connected layer with 32 input features and 7 output features

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        # x = x.view(-1, 7)
        return x
    
# Define the Signed Distance Function MLP model
class SDFModel_min_dist_overall(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(SDFModel_min_dist_overall, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)   # Fully connected layer with 7 input features and 32 output features
        self.relu = nn.ReLU()         # ReLU activation function
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)  # Fully connected layer with 32 input features and 16 output features
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)  # Fully connected layer with 16 input features and 21 output features

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = x.view(-1, 7)
        return x
    
class SDF_Dataset_Value(Dataset):   #this class handles a single conf in custom pybullet env
    def __init__(self, dists, jnt_states, n_conf = 50000, is_train = True, transform = None):
        # self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        self.n_conf = n_conf
        self.dists = dists  #(N, num_joints, num_obstacles)
        self.jnt_states = jnt_states  #(N, num_joints)

    def __len__(self):
        return self.n_conf

    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.to_list()

        #get dist of every joint from each obstacle of a particular conf #(7, 3)
        dists = self.dists[idx]
        jnt_states = self.jnt_states[idx]

        return jnt_states, dists

def get_dataset(file):
    dists = np.load(file)
    # print('dists shape: {}'.format(dists.shape))
    return dists

def train_sdf_fn(args, dataset):
    training_start_time = time.time()

    # change action dim to 2 later
    input_dim = 7    # no of joints  (7)
    hidden_dim_1 = 32   #
    hidden_dim_2 = 128
    hidden_dim_3 = 64
    # output_dim = 21   # 7*3
    # output_dim = 21
    output_dim = 7  # (for sdf fn per joint)
 
    #load model architecture 
    # sdf_model = SDFModel_min_dist_per_joint(input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim)
    sdf_model = SDFModel(input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim)
    sdf_model = sdf_model.to(device)

    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#

    trainer = Trainer_jayaram(sdf_model, dataset)

    for i in range(args.epochs):
        print(f'Epoch {i} / {args.epochs} | {args.save_ckpt}')
        trainer.train(device, i, n_train_steps=args.number_of_steps_per_epoch)

    # Save results
    # if save_results:
    #     # Rename the final training log instead of re-writing it
    #     training_log_path = os.path.join(args.output_dir, "training_log.pkl")
    #     os.rename(epoch_training_log_path, training_log_path)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
    print("Done.")
    print("")
    print("Total training time: {} seconds.".format(time.time() - training_start_time))
    print("")


if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument(
    #     "-i", "--input-data-path", help="Path to training data."
    # )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Path to output directory for training results. Nothing specified means training results will NOT be saved.",
    )

    parser.add_argument(
        "-save_ckpt",
        "--save_ckpt",
        type=str,
        default="/home/jayaram/research/research_tracks/table_top_rearragement/Pybullet_Custom/logs/sdf_single_scene_3_spheres",
        help="save checkpoints of diffusion model while training",
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=150, help="Number of epochs to train."
    )

    parser.add_argument(
        "-n_steps_per_epoch",
        "--number_of_steps_per_epoch",
        type=int,
        default=10000,
        help="Number of steps per epoch",
    )

    parser.add_argument(
        "-hr",
        "--horizon",
        type=int,
        default=50,
        help="Horizon or no of waypoints in path",
    )

    # parser.add_argument(
    #     "-b",
    #     "--batch-size",
    #     type=int,
    #     required=True,
    #     help="The number of samples per batch used for training.",
    # )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0002,
        help="The learning rate used for the optimizer.",
    )

    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=8,
        help='The number of subprocesses ("workers") used for loading the training data. 0 means that no subprocesses are used.',
    )

    # parser.add_argument(
    #     "-rt",
    #     "--retrain-network",
    #     type=str2bool,
    #     default=False,
    #     help="Retraines network after unfreezing last few modules.",
    # )

    args = parser.parse_args()

    sdf_file = "/home/jayaram/research/research_tracks/table_top_rearragement/Pybullet_Custom/sdf_dists_three_spheres_min_distance_per_joint.npy"
    dists = get_dataset(sdf_file)

    jnt_states_file = "/home/jayaram/research/research_tracks/table_top_rearragement/Pybullet_Custom/joint_states.npy"
    jnt_states = get_dataset(jnt_states_file)

    sdf_dataset = SDF_Dataset_Value(dists, jnt_states, n_conf = 50000)

    # Train the network on many samples
    train_sdf_fn(args, sdf_dataset)