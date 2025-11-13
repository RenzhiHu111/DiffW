import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet
import math
from tensorboardX import SummaryWriter
import kornia.losses

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def noise_estimation_loss(model, x0, t, e, b, message, epoch, combined_attack):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:6, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output_0, output_noise, output_message = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float(), message)
    criterion1 = nn.MSELoss()
    x0_loss = criterion1(output_0, x0[:, 3:6, :, :])
    criterion2 = nn.MSELoss()
    message_loss = criterion2(output_message, message)
    decoded_rounded = output_message.detach().cpu().numpy().round().clip(0, 1)
    bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
            x0.shape[0] * message.shape[1])

    psnr = kornia.losses.psnr_loss(output_0.detach(), x0[:, 3:6, :, :], 2)

    output = (x - output_0 * a.sqrt()) / (1.0 - a).sqrt()

    # Combined Attack Training Strategy
    if combined_attack:
        sum = x0_loss + 10 * message_loss
    else:
    # Single Attack Training Strategy
        if epoch > 5:
            sum = 10 * x0_loss + message_loss
        else:
            sum = x0_loss + 10 * message_loss


    return x0_loss, message_loss, bitwise_avg_err, sum, psnr

class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = DiffusionUNet(config)

        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        writer = SummaryWriter('./logs/log')
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):

                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:6, :, :])
                message = torch.Tensor(np.random.choice([0, 1], (x[:, 3:6, :, :].shape[0], self.config.watermarking.message_length))).to('cuda')
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                x0_loss,  message_loss, bitwise_avg_err, all_loss, psnr = noise_estimation_loss(self.model, x, t, e, b, message, epoch, self.config.watermarking.combined_attack)
                loss = all_loss
                writer.add_scalar('message_loss', message_loss, epoch)
                writer.add_scalar('x0_loss', x0_loss, epoch)
                writer.add_scalar('all_loss', all_loss, epoch)

                if self.step % 10 == 0:
                    print(f"step: {self.step}, x0_loss: {x0_loss.item()}, message_loss: {message_loss.item()}, bitwise_avg_err: {bitwise_avg_err.item()}, "
                          f"loss: {loss.item()}, psnr: {psnr.item()}, data time: {data_time / (i + 1)}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_ddpm'))

        writer.close()

    def sample_image(self, x_cond, x, message, last=True, patch_locs=None, patch_size=None):
        # non-uniform sampling
        start_value = self.config.diffusion.num_diffusion_timesteps-1
        num_samples = self.args.sampling_timesteps
        samples = []
        start_tolerance = 1
        tolerance = (start_value / (num_samples - start_tolerance) - start_tolerance) / (num_samples - start_tolerance)
        for i in range(num_samples):
            samples.append(start_value - i * start_tolerance)
            start_tolerance += tolerance
        samples = [math.floor(value) for value in samples]
        samples.reverse()
        seq = samples
        xs = utils.sampling.generalized_steps(x, x_cond, seq, message, self.model, self.betas, eta=0.)
        return xs

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)

            message = torch.Tensor(np.random.choice([0, 1], (x_cond.shape[0], self.config.watermarking.message_length))).to('cuda')

            x = self.sample_image(x_cond, x, message)
            x1 = x[0][-1]
            x1 = inverse_data_transform(x1)

            message1 = x[2][-1]

            decoded_rounded = message1.detach().cpu().numpy().round().clip(0, 1)
            message_detached = message.detach().cpu().numpy()
            print('original: {}'.format(message_detached))
            print('decoded : {}'.format(decoded_rounded))
            print('error : {:.3f}'.format(np.mean(np.abs(decoded_rounded - message_detached))))

            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.logging.save_image(x1[i], os.path.join(image_folder, str(step), f"{i}.png"))
