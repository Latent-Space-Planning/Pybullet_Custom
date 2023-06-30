import h5py
import numpy as np
import copy
import torch
from scripts.diffusion_functions import *
import einops

class Traj_train_dataset:
    def __init__(self, dataset_path, n_diffusion_steps=255) -> None:
        _data = h5py.File(dataset_path, 'r')
        self._data_global = _data['global_solutions']
        self._data_hybrid = _data['hybrid_solutions']

        self.num_traj_gl = self._data_global.shape[0]
        self.num_traj_hyb = self._data_hybrid.shape[0]
        self.n_diffusion_steps = 255
        self.traj_len = 50
        self.state_size = 7


    def sample_trajectories(self, batch_size=2048):
        '''Sample 50% of the samples from global solutions (approx. 3M) and 50% from hybrid solutions (approx. 3M)
        '''
        sample_set1 = self._data_global[np.sort(np.random.choice(self.num_traj_gl, batch_size//2, replace=False)), :, :]
        np.random.shuffle(sample_set1)
        sample_set2 = self._data_hybrid[np.sort(np.random.choice(self.num_traj_hyb, size=batch_size//2, replace=False)), :, :]
        np.random.shuffle(sample_set2)

        return torch.tensor(einops.rearrange(np.concatenate((sample_set1, sample_set2), axis=0), 'b n c -> b c n'))
    
    def generate_training_batch(self, batch_size=2048):

        # Refer to the Training Algorithm in Ho et al 2020 for the psuedo code of this function
            
        x0 = self.sample_trajectories(batch_size=batch_size).numpy()
        time_step = np.random.randint(1, self.n_diffusion_steps+1, size = (batch_size, ))

        # Remember, for each channel, we get a multivariate normal distribution.
        mean = np.zeros(self.traj_len)
        cov = np.eye(self.traj_len)
        eps = np.random.multivariate_normal(mean, cov, (batch_size, 7))

        beta = schedule_variance(self.n_diffusion_steps)
        alpha = 1 - beta
        alpha_bar = np.reshape(np.array(list(map(lambda t:np.prod(alpha[:t]), time_step))), (-1, 1, 1))  # Tested: This works
        
        # Size chart:
        # x0         => (num_samples, 2, traj_len)
        # xt         => (num_samples, 2, traj_len)
        # alpha_bar  => (num_samples, 1, 1)
        # eps        => (num_samples, 2, traj_len)
        
        xt = (np.sqrt(alpha_bar) * x0) + (np.sqrt(1 - alpha_bar) * eps)

        # CONDITIONING:
        xt[:, :, 0] = x0[:, :, 0].copy()
        xt[:, :, -1] = x0[:, :, -1].copy()

        X = torch.tensor(xt, dtype = torch.float32)
        Y = torch.tensor(eps, dtype = torch.float32)

        return X, Y, torch.tensor(time_step)


