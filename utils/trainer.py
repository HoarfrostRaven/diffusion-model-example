import torch
import torch.nn.functional as F
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model, dataloader, optim, timesteps, n_epoch, lrate, device, beta1=1e-4, beta2=0.02, context_enable=False, save_dir=None):
        self.model = model
        self.dataloader = dataloader
        self.optim = optim
        self.timesteps = timesteps
        self.n_epoch = n_epoch
        self.lrate = lrate
        self.device = device
        self.context_enable = context_enable
        self.save_dir = save_dir

        # construct DDPM noise schedule
        b_t = (beta2 - beta1) * torch.linspace(0, 1,
                                               timesteps + 1, device=device) + beta1
        a_t = 1 - b_t
        self.ab_t = torch.cumsum(a_t.log(), dim=0).exp()
        self.ab_t[0] = 1

    # helper function: perturbs an image to a specified noise level
    def perturb_input(self, x, t, noise):
        return self.ab_t.sqrt()[t, None, None, None] * x + (1 - self.ab_t[t, None, None, None]) * noise

    def train(self):
        # set into train mode
        self.model.train()

        for ep in range(self.n_epoch):
            print(f'epoch {ep}')

            # linearly decay learning rate
            self.optim.param_groups[0]['lr'] = self.lrate * \
                (1 - ep / self.n_epoch)

            pbar = tqdm(self.dataloader, mininterval=2)
            for x, c in pbar:  # x: images  c: context
                self.optim.zero_grad()
                x = x.to(self.device)

                if self.context_enable:
                    c = c.to(x)
                    # randomly mask out c
                    context_mask = torch.bernoulli(
                        torch.zeros(c.shape[0]) + 0.9).to(self.device)
                    c = c * context_mask.unsqueeze(-1)

                # perturb data
                noise = torch.randn_like(x)
                t = torch.randint(1, self.timesteps + 1,
                                  (x.shape[0],)).to(self.device)
                x_pert = self.perturb_input(x, t, noise)

                # use network to recover noise
                if self.context_enable:
                    pred_noise = self.model(x_pert, t / self.timesteps, c=c)
                else:
                    pred_noise = self.model(x_pert, t / self.timesteps)

                # loss is mean squared error between the predicted and true noise
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()

                self.optim.step()

            # save model periodically
            if self.save_dir is not None and (ep % 4 == 0 or ep == int(self.n_epoch - 1)):
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                torch.save(self.model.state_dict(),
                           f"{self.save_dir}context_model_{ep}.pth")
                print(f"saved model at {self.save_dir}context_model_{ep}.pth")
