import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os

class Denoiser(nn.Module):

    def __init__(self, model_name, x_dim, time_emb_dim, channel_dims = (256, 128, 64, 32)):

        super(Denoiser, self).__init__()

        # self.T = timesteps
        # self.beta = torch.tensor(beta, dtype = torch.float32)
        # self.alpha = 1 - self.beta
        
        #-------------------Architecture-----------------------#
        
        self.in_layer = nn.Linear(x_dim, channel_dims[0])
        self.in_time_mlp = nn.Linear(time_emb_dim, channel_dims[0])
        self.in_batch_norm = nn.BatchNorm1d(channel_dims[0])

        self.layers = []
        self.time_mlps = []
        self.batch_norms = []
        for dim in range(1, len(channel_dims)):
            self.layers.append(nn.Linear(channel_dims[dim-1], channel_dims[dim]))
            self.time_mlps.append(nn.Linear(time_emb_dim, channel_dims[dim]))
            self.batch_norms.append(nn.BatchNorm1d(channel_dims[dim]))
        for dim in range(len(channel_dims)-1, 0, -1):
            self.layers.append(nn.Linear(channel_dims[dim], channel_dims[dim-1]))
            self.time_mlps.append(nn.Linear(time_emb_dim, channel_dims[dim-1]))
            self.batch_norms.append(nn.BatchNorm1d(channel_dims[dim-1]))

        self.out_layer = nn.Linear(channel_dims[0], x_dim)

        #self.batchnorm = nn.BatchNorm1d(2048)

        #-----------------------------------------------------#

        self.model_name = model_name
        self.getPositionEncoding(time_emb_dim, time_emb_dim)
        self.model = nn.ModuleList(self.layers)
        self.time_model = nn.ModuleList(self.time_mlps)
        self.batch_norm_model = nn.ModuleList(self.batch_norms)
        
        if not os.path.exists(model_name):
            os.mkdir(model_name)
            self.losses = np.array([])
        else:
            self.load()

        #self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001)
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, X, time_emb):

        Y = self.in_layer(X)
        Y = F.relu(Y)
        Y = self.in_batch_norm(Y)
        
        h = self.in_time_mlp(time_emb)
        h = F.relu(h)

        for i in range(len(self.layers)):

            Y = self.layers[i](Y+h)
            Y = self.batch_norms[i](Y)
            Y = F.relu(Y)

            h = self.time_mlps[i](time_emb)
            h = F.relu(h)
        
        Y = self.out_layer(Y+h)
        
        return Y
    
    def getPositionEncoding(self, T, embed_dim, n = 10000):

        # Initialize Positional Encoding Matrix
        self.P = np.zeros((T, embed_dim))

        for k in range(T):
            for i in np.arange(int(embed_dim/2)):
                denominator = np.power(n, 2*i/embed_dim)    
                self.P[k, 2*i] = np.sin(k/denominator)
                self.P[k, 2*i+1] = np.cos(k/denominator)
    
    def loss_fn(self, Y_pred, Y_true):

        with torch.no_grad():    
            loss = nn.L1Loss()(Y_pred, Y_true).item()
        if loss < 1:
            output_loss = nn.MSELoss()
        else:
            output_loss = nn.L1Loss()
        
        return 10*output_loss(Y_pred, Y_true)
    
    def save(self):

        torch.save(self.state_dict(), self.model_name + "/weights_latest.pt")
        np.save(self.model_name + "/losses.npy", self.losses)

    def save_checkpoint(self, checkpoint):
        
        torch.save(self.state_dict(), self.model_name + "/weights_" + str(checkpoint) + ".pt")
        np.save(self.model_name + "/latest_checkpoint.npy", checkpoint)
    
    def load(self):

        self.losses = np.load(self.model_name + "/losses.npy")
        self.load_state_dict(torch.load(self.model_name + "/weights_latest.pt"))
        print("Loaded Model at " + str(self.losses.size) + " epochs")

    def load_checkpoint(self, checkpoint):

        _ = input("Press Enter if you are running the model for inference, or Ctrl+C\n(Never load a checkpoint for training! This will overwrite progress)")
        
        latest_checkpoint = np.load(self.model_name + "/latest_checkpoint.npy")
        self.load_state_dict(torch.load(self.model_name + "/weights_" + str(checkpoint) + ".pt"))
        self.losses = np.load(self.model_name + "/losses.npy")[:checkpoint]