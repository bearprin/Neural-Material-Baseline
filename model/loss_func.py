import numpy as np
import torch

def mu_transform(x, mu):
    # return torch.log1p(x)
    x = torch.log1p(mu * x) / np.log1p(mu)
    return x


def inv_mu_transform(x, mu):
    # return torch.expm1(x)
    x = torch.expm1(x * np.log1p(mu)) / mu
    return x


def brdf_to_rgb(wi, wo, brdf, io=True):
    if wo.dim() == 3:
        B, N, C = wo.shape
        wo = wo.reshape(-1, 3)
        wi = wi.unsqueeze(1).expand(-1, N, -1).reshape(-1, wi.shape[-1])
    if io:
        wiz = wi[:, 2:3]
        woz = wo[:, 2:3]

    # else:
    #     hx = torch.reshape(rvectors[:, 0], (-1, 1))
    #     hy = torch.reshape(rvectors[:, 1], (-1, 1))
    #     hz = torch.reshape(rvectors[:, 2], (-1, 1))
    #     dx = torch.reshape(rvectors[:, 3], (-1, 1))
    #     dy = torch.reshape(rvectors[:, 4], (-1, 1))
    #     dz = torch.reshape(rvectors[:, 5], (-1, 1))
    #
    #     theta_h = torch.atan2(torch.sqrt(hx ** 2 + hy ** 2), hz)
    #     theta_d = torch.atan2(torch.sqrt(dx ** 2 + dy ** 2), dz)
    #     phi_d = torch.atan2(dy, dx)
    #     wiz = torch.cos(theta_d) * torch.cos(theta_h) - \
    #           torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
    rgb = brdf * torch.clamp(woz, 0, 1)
    return rgb


def reverse_huber_loss(y_pred, y_true, t=0.1):
    abs_diff = torch.abs(y_pred - y_true)
    squared_diff = (y_pred - y_true) ** 2
    # Condition: |T(f) - T(f_s)| <= t
    l1_loss = abs_diff
    # Condition: |T(f) - T(f_s)| > t
    l2_loss = (squared_diff + t ** 2) / (2 * t)
    # Apply the conditions
    loss = torch.where(abs_diff <= t, l1_loss, l2_loss)
    return loss.mean()


def mape_loss(y_true, y_pred, t=0.01):
    l1 = torch.abs(y_true - y_pred)
    l1 = l1 / (y_true.abs().detach() + t)
    return l1.mean()