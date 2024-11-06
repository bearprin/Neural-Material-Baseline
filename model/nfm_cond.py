import torch
import torch.nn as nn
import normflows as nf


class NFMCond(nn.Module):
    def __init__(self, layers=2, context_dim=32):
        super().__init__()
        self.context_encoder = nf.nets.MLP(layers=[2, 32, context_dim], init_zeros=True)

        self.base = nf.distributions.base.DiagGaussian(2, trainable=True)

        latent_size = 2
        hidden_units = 32
        hidden_layers = 2
        flows = list()
        for i in range(layers):
            flows.append(nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units,
                                                              num_context_channels=context_dim))
            flows.append(nf.flows.LULinearPermute(2))

        self.model = nf.ConditionalNormalizingFlow(self.base, flows)

    def sample(self, wi):
        context = self.context_encoder(wi[:, :2])
        z, log_pdf = self.model.sample(num_samples=wi.shape[0], context=context)
        return z, log_pdf

    def forward_kld(self, wi, wo):
        context = self.context_encoder(wi[:, :2])
        if wo.dim() == 3:
            B, N, C = wo.shape
            wo = wo.reshape(-1, 3)
            context = context.unsqueeze(1).expand(-1, N, -1).reshape(-1, context.shape[-1])
        wo = wo[:, :2]
        return self.model.forward_kld(wo, context=context)

    def log_prob(self, wi, wo):
        context = self.context_encoder(wi[:, :2])
        if wo.dim() == 3:
            B, N, C = wo.shape
            wo = wo.reshape(-1, 3)
            context = context.unsqueeze(1).expand(-1, N, -1).reshape(-1, context.shape[-1])
        wo = wo[:, :2]  # only use x, y for disk
        return self.model.log_prob(wo, context=context)

if __name__ == '__main__':
    model = NFMCond()
    wi = torch.randn(10, 3)
    wo = torch.randn(10, 5, 3)
    print(model.forward_kld(wi, wo))