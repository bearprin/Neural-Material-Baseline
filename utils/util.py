import torch

def generate_query_angles(wi=[0, 0, 1], resolution=512, device='cuda'):
    wi = torch.tensor(wi, dtype=torch.float32, device=device).view(1, -1)
    wox = torch.linspace(-1, 1, resolution, device=device)
    woy = torch.linspace(-1, 1, resolution, device=device)
    wox, woy = torch.meshgrid(wox, woy)
    woz_squared = 1 - wox ** 2 - woy ** 2
    woz = torch.sqrt(torch.clamp(woz_squared, min=0))
    wo = torch.stack((wox, woy, woz), dim=-1).reshape(-1, 3)
    wi = wi.repeat(wo.shape[0], 1)

    angles = torch.cat([wi, wo], dim=-1)
    mask = woz <= 0
    return angles, mask