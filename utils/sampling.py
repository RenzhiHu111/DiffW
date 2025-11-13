import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop
import numpy as np

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

# When eta=1 it is DDPM; when eta = 0, it is DDIM.
def generalized_steps(x, x_cond, seq, message, model, b, eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        message_all = []
        xs = [x]
        xs_n = [x]
        x_noise = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            xt_n = xs_n[-1].to('cuda')

            encoded_image, noised_image, decoded_message = model(torch.cat([x_cond, xt], dim=1), t, message)
            et = (xt - encoded_image * at.sqrt()) / (1 - at).sqrt()
            x0_t = encoded_image
            x0_preds.append(x0_t.to('cpu'))

            x_noise1 = noised_image
            x_noise.append(x_noise1.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

            et_n = (xt_n - noised_image * at.sqrt()) / (1 - at).sqrt()
            x0_t_n = noised_image

            c1_n = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2_n = ((1 - at_next) - c1_n ** 2).sqrt()
            xt_next_n = at_next.sqrt() * x0_t_n + c1_n * torch.randn_like(x) + c2_n * et_n
            xs_n.append(xt_next_n.to('cpu'))

            message_all.append(decoded_message.to('cpu'))

    return xs, x0_preds, message_all, x_noise
