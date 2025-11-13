import torch
import torch.nn as nn
import utils
import torchvision
import os
import numpy as np

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='snow', r=None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        total = 0
        count = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                print(f"starting processing from image {y}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_cond = data_transform(x_cond)
                message = torch.Tensor(np.random.choice([0, 1], (x_cond.shape[0], self.config.watermarking.message_length))).to('cuda')

                x = torch.randn(x_cond.size(), device=self.diffusion.device)

                x_output = self.diffusion.sample_image(x_cond, x, message)

                x2 = x_output[1][-1]
                x2 = inverse_data_transform(x2)
                message1 = x_output[2][-1]
                x_n = x_output[3][-1]
                x_n = inverse_data_transform(x_n)

                decoded_rounded = message1.detach().cpu().numpy().round().clip(0, 1)
                message_detached = message.detach().cpu().numpy()
                print('original: {}'.format(message_detached))
                print('decoded : {}'.format(decoded_rounded))
                print('error : {:.3f}'.format(np.mean(np.abs(decoded_rounded - message_detached))))
                total += np.mean(np.abs(decoded_rounded - message_detached))
                count += 1
                utils.logging.save_image(x2, os.path.join(image_folder, f"{y[0]}.png"))

            print('total : {:.6f}'.format((total / count)*100))
            print('total1 : {:.6f}'.format(100-(total / count) * 100))

    def diffusive_restoration(self, x_cond, message, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, message, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
