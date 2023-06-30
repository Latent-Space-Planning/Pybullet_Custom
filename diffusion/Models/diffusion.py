import torch
import torch.nn as nn
import numpy as np

class DiffusionModel(nn.Module):

    def __init__(self, channels, length, timesteps):

        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.L = length
        self.C = channels
        self.T = timesteps

        beta = self.variance_schedule()         
        alpha = 1. - beta
        alpha_cumprod = torch.cumprod(alpha, axis=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]])

        # Basic Parameters:
        self.register_buffer('beta', beta)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('alpha_cumprod_prev', alpha_cumprod_prev)

        # Parameters to get q_sample = q(x_t | x_0):
        self.register_buffer('sqrt_alpha', torch.sqrt(alpha))
        self.register_buffer('sqrt_one_minus_alpha', torch.sqrt(1. - alpha))

        # Parameters to get q_sample = q(x_t | x_{t-1}):
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1. - alpha_cumprod))
        
        # Parameters to get p(x_0 | x_T):
        self.register_buffer('sqrt_recip_alpha', torch.sqrt(1. / alpha))
        self.register_buffer('sqrt_recip_alpha_cumprod', torch.sqrt(1. / alpha_cumprod))
        self.register_buffer('sqrt_recip_one_minus_alpha_cumprod', torch.sqrt(1. / (1 - alpha_cumprod)))

    def variance_schedule(self, s = 0.008):

        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = self.T + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)

        return torch.tensor(betas_clipped, dtype=torch.float32)
    
    def forward_diffuse(self, x0):
        """
        x0 => Noiseless data at timestep 0 of shape (channels x length)
        """

        X = torch.empty(size = (self.T + 1, self.C, self.L))
        X[0] = x0.clone()

        noise = torch.randn(size = (self.T, self.C, self.L))

        # Loop from 1 to T (T included)
        for t in range(1, self.T + 1):

            # Beta and all its counterparts are of size 255, with indices 0 -> 254.
            # This means that the beta for timestep 1 is beta[0], so for any t, beta is beta[t-1]
            
            X[t] = self.sqrt_alpha[t-1] * X[t-1] + self.sqrt_one_minus_alpha[t-1] * noise[t-1]

            X[t, :, 0] = x0[0, :, 0].clone()
            X[t, :, -1] = x0[0, :, -1].clone()

        return X

    def q_sample(self, x0, t):

        noise = torch.randn_like(x0).to("cuda").to(self.device)
        a = torch.reshape(self.sqrt_alpha_cumprod[t-1], (x0.shape[0], 1, 1)).to(self.device)
        b = torch.reshape(self.sqrt_one_minus_alpha_cumprod[t-1], (x0.shape[0], 1, 1)).to(self.device)

        # Beta and all its counterparts are of size 255, with indices 0 -> 254.
        # This means that the beta for timestep 1 is beta[0], so for any t, beta is beta[t-1]

        xt = a * x0 + b * noise

        xt[0, :, 0] = x0[0, :, 0].clone()
        xt[0, :, -1] = x0[0, :, -1].clone()

        return xt, noise
    
    def denoise(self, xt, t, denoiser):

        if t > 1:
            z = np.reshape(np.random.multivariate_normal(mean = np.zeros((self.C * self.L,)), cov = np.eye(self.C * self.L)), (1, self.C, self.L))
        else:
            z = np.zeros((self.C, self.L))
        z = torch.tensor(z, dtype = torch.float32).to(self.device)

        time_in = torch.tensor([t], dtype = torch.float32).to(self.device)
        xt = xt.float().to(self.device)

        noise = denoiser(xt, time_in)

        xt_prev = self.sqrt_recip_alpha[t-1] * (xt - (self.sqrt_one_minus_alpha[t-1] / self.sqrt_one_minus_alpha[t-1]) * noise) + self.beta[t-1]*z

        xt_prev[0, :, 0] = xt[0, :, 0].clone()
        xt_prev[0, :, -1] = xt[0, :, -1].clone()
        
        return xt_prev
