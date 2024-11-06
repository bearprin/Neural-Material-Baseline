import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from .loss_func import inv_mu_transform


class MLP(torch.nn.Module):
    """
    Neural BRDF model
    """

    def __init__(self, layer_dims=None, weight_norm=False, mu=32):
        super(MLP, self).__init__()
        self.is_encoding = False
        if layer_dims is None:
            layer_dims = [6, 32, 32, 3]
        self.layer_dims = layer_dims
        self.weight_norm = weight_norm

        # Create a list to hold all layers
        self.layers = torch.nn.ModuleList()

        # Add all layers specified in layer_dims
        for i in range(len(layer_dims) - 1):
            self.layers.append(torch.nn.Linear(in_features=layer_dims[i], out_features=layer_dims[i + 1], bias=True))

        # Initialize weights
        self._initialize_weights()

        self.post_process = nn.Identity() if mu <= 0 else lambda x: F.relu(inv_mu_transform(x, mu))

    def _initialize_weights(self):
        for layer in self.layers[:-1]:
            torch.nn.init.zeros_(layer.bias)
            torch.nn.init.xavier_uniform_(layer.weight)
            if self.weight_norm:
                weight_norm(layer)


    def forward(self, wi, wo):
        if wo.dim() == 3:
            B, N, C = wo.shape
            wi = wi.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([wi, wo], dim=-1).reshape(-1, self.layer_dims[0])
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x), 0.01)
        x = self.post_process(self.layers[-1](x))
        return x
