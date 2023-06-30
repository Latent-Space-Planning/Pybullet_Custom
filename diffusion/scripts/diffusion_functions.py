import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import einops

def cosine_func(t, T):
    
    s = 1e-10
    
    return np.cos((t/T + s) * (np.pi/2) / (1 + s)) ** 0.15

def schedule_variance(T, thresh = 0.02):
    
    return np.linspace(0, thresh, T+1)[1:]

def plot_trajectories(x):

    for i in range(x.shape[0]):
        plt.scatter(x[i, 0, 1:-1], x[i, 1, 1:-1])
        plt.scatter(x[i, 0, 0], x[i, 1, 0], color = 'black')
        plt.scatter(x[i, 0, -1], x[i, 1, -1], color = 'red')

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

def pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * (((x - mu) / sigma) ** 2))

def KL_divergence(distr1, distr2):

    return np.mean(distr1 * np.log(distr1 / distr2))

def KL_divergence_against_gaussian(sample):

    mu = np.mean(sample)
    sigma = np.var(sample)

    x = np.linspace(-10, 10, int(1e+6))

    distr1 = pdf(x, 0, 1) 
    distr2 = pdf(x, mu, sigma)

    return KL_divergence(distr1, distr2)

def forward_diffuse(trajectory, T, beta):

    # Remember that the trajectory that is being input here is a singular starting trajectory of shape (n+2, 2)
    
    # Gather information from trajectory:
    n = trajectory.shape[0] - 2

    alpha = 1 - beta
    
    # Initialize the diffusion trajectories across time steps:
    diffusion_trajs = np.zeros((T+1, n+2, 2))
    diffusion_trajs[0] = trajectory.copy()

    # Calculate epsilons to forward diffuse current trajectory:
    mean = np.zeros(n*2)
    cov = np.eye(n*2)
    eps = np.random.multivariate_normal(mean, cov, size = (T,)).reshape(T, n, 2)
    kl_divs = np.zeros((T, ))

    for i in range(1, T+1):

        diffusion_trajs[i, 1:-1] = np.sqrt(alpha[i-1]) * diffusion_trajs[i-1, 1:-1] + np.sqrt(1 - alpha[i-1]) * eps[i-1]
        diffusion_trajs[i, 0] = trajectory[0].copy()
        diffusion_trajs[i, -1] = trajectory[-1].copy()
        kl_divs[i-1] = KL_divergence_against_gaussian(diffusion_trajs[i].flatten())

    return diffusion_trajs.copy(), eps.copy(), kl_divs.copy()

def reverse_diffuse(xT, eps, T):

    #xT = np.random.multivariate_normal(mean = np.zeros((traj_len*2,)), cov = np.eye(traj_len*2)).reshape(traj_len, 2)

    diffusion_trajs = np.zeros((T+1, xT.shape[0], 2))
    diffusion_trajs[T] = xT.copy()

    beta = schedule_variance(T)
    alpha = 1 - beta

    for t in range(T, 0, -1):

        diffusion_trajs[t-1] = (diffusion_trajs[t] - np.sqrt(1 - alpha[t-1]) * eps[t-1]) / np.sqrt(alpha[t-1])

    return diffusion_trajs.copy()

def train_denoiser(model, num_samples, traj_len, T, epochs_per_set = 10, num_sets = 10):

    losses = np.zeros((num_sets, epochs_per_set))

    for s in range(num_sets):

        X = np.zeros((num_samples*model.T, 2*(traj_len+2)))
        Y_true = np.zeros((num_samples*model.T, 2*(traj_len)))

        trajectories = generate_trajectories(num_samples, traj_len)

        for i in range(num_samples):

            print("Generating Sample ", i+1)

            # Diffuse just one sample trajectory:
            diff_traj, epsilon = forward_diffuse(trajectories[i], model.T, model.beta)

            clear_output(wait=True)

            X[i*T : (i+1)*T] = diff_traj[1:].reshape(T, 2*(traj_len+2))
            Y_true[i*T : (i+1)*T] = epsilon.reshape(T, 2*(traj_len))

        X = torch.tensor(X, dtype = torch.float32)
        Y_true = torch.tensor(Y_true, dtype = torch.float32)

        for e in range(epochs_per_set):
                
            model.train(True)

            Y_pred = model(X)

            model.optimizer.zero_grad()

            loss = model.loss_fn(Y_pred, Y_true)

            loss.backward()

            model.optimizer.step()

            loss_value = torch.norm(Y_pred - Y_true).item()
            losses[s, e] = loss_value/epochs_per_set

            print("Current epoch loss = ", loss_value)

            clear_output(wait=True)

        model.save()
        np.save("Models/" + model.model_name + "/losses.npy", losses)

    return losses

def length_gradient(X):

    grad = np.zeros(X.shape)

    for i in range(1, X.shape[2] - 1):

        grad[:, :, i] = 2 * (2 * X[:, :, i] - (X[:, :, i-1] + X[:, :, i+1]))

    return grad

def generate_training_sample(num_samples, generate_trajectories, traj_len = 1000, T = 500, bounds = np.array([[-1.5, -1.5], [1.5, 1.5]])):

    # Refer to the Training Algorithm in Ho et al 2020 for the psuedo code of this function
        
    x0 = generate_trajectories(num_samples = num_samples, traj_len = traj_len, bounds = bounds).numpy()
    time_step = np.random.randint(1, T+1, size = (num_samples, ))

    # Remember, for each channel, we get a multivariate normal distribution.
    mean = np.zeros(traj_len)
    cov = np.eye(traj_len)
    eps = np.random.multivariate_normal(mean, cov, (num_samples, 2))

    beta = schedule_variance(T)
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

