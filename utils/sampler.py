import torch
import numpy as np


class Sampler:
    def __init__(self, model, timesteps, width, height, device, beta1=1e-4, beta2=0.02):
        self.model = model
        self.timesteps = timesteps
        self.width = width
        self.height = height
        self.device = device
        # construct DDPM noise schedule
        self.b_t = (beta2 - beta1) * torch.linspace(0, 1,
                                                    timesteps + 1, device=device) + beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()
        self.ab_t[0] = 1

    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
    def denoise_ddpm(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) /
                (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        return mean + noise

    # define sampling function for DDIM
    # removes the noise using ddim
    def denoise_ddim(self, x, t, t_prev, pred_noise):
        ab = self.ab_t[t]
        ab_prev = self.ab_t[t_prev]

        x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise

        return x0_pred + dir_xt

    # sample using standard algorithm
    @torch.no_grad()
    def sample_ddpm(self, n_sample=1, n=20, context=None):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, self.width,
                              self.height).to(self.device)
        # array to keep track of generated steps for plotting
        intermediate = []

        for i in range(self.timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / self.timesteps])[:, None,
                                                   None, None].to(self.device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            # predict noise e_(x_t,t, ctx)
            eps = self.model(samples, t, c=context)
            samples = self.denoise_ddpm(samples, i, eps, z)

            if i % n == 0 or i == self.timesteps or i < 8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate

    # sample quickly using DDIM
    @torch.no_grad()
    def sample_ddim(self, n_sample=1, n=20, context=None):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, self.width,
                              self.height).to(self.device)
        # array to keep track of generated steps for plotting
        intermediate = []
        step_size = self.timesteps // n

        for i in range(self.timesteps, 0, -step_size):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / self.timesteps])[:, None,
                                                   None, None].to(self.device)

            eps = self.model(samples, t, c=context)
            samples = self.denoise_ddim(samples, i, i - step_size, eps)
            intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate
