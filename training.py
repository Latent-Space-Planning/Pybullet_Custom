import os
import copy
import numpy as np
import torch
import einops
import pdb
import os
from timer import Timer
from arrays import batch_to_device, to_np, apply_dict, to_device
import torch.nn as nn
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer_jayaram(object):   #/home2/jayaram.reddy/test_diffusion_planning/logs/maze2d-test/diffusion
    def __init__(
        self,
        sdf_model,
        dataset,  
        # renderer, 
        ema_decay=0.995,
        train_batch_size=128,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=40000,
        save_parallel=False,
        # results_folder='/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/logs/maze2d-test/diffusion',
        results_folder='/home/jayaram/research/research_tracks/table_top_rearragement/Pybullet_Custom/logs/sdf_single_scene_3_spheres',
        n_reference=50,
        n_samples=10,
        bucket=None,
    ):
        super().__init__()
        self.model = sdf_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        # self.renderer = renderer
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        # self.dataloader_vis = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        # ))
        self.optimizer = torch.optim.Adam(sdf_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples
        self.criterion = nn.MSELoss()  # Mean Squared Error loss
        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, device, epoch_no, n_train_steps):
        isExist = os.path.exists(self.logdir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.logdir)
            print("The new directory is created!")  
            
        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)    #batch[0].shape: (128, 7), batch[1].shape: (128, 7, 3)
                # batch[1] = batch[1].view(-1, 21)
                # batch = batch_to_device(batch)
                for i, el in enumerate(batch):
                    batch[i] = to_device(batch[i])

                # path, cond = self.dataset[0][0], self.dataset[0][1]    #for single path training
                # loss = self.model.loss(path, cond, device)

                #for value fn training, infos is removed
                # loss, infos = self.model.loss(*batch)
                outputs = self.model(batch[0])
                targets = batch[1]
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulate_every
                wandb.log({'loss': loss, 'epoch': epoch_no, 'step no': step}) #, 'batch': t})
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_freq == 0:
                # infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                # print(f'{self.step}: {loss:8.4f} : {infos_str} | t: {timer():8.4f}')

                print(f'{self.step}: {loss:8.4f}  | t: {timer():8.4f}')

            # if self.step == 0 and self.sample_freq:
            #     self.render_reference(self.n_reference)

            # if self.sample_freq and self.step % self.sample_freq == 0:
            #     self.render_samples(n_samples=self.n_samples)

            self.step += 1

        label = epoch_no
        self.save(label)

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        isExist = os.path.exists(self.logdir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.logdir)
            print("The new directory is created!")        

        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


