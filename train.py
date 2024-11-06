import argparse
import time
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

import torch
import torch.nn.functional as F
from lightning import seed_everything

from model.MLP import MLP
from model import loss_func, nfm_cond
from dataset.sample_dataset import SampleDataset
from utils import generate_query_angles

seed_everything(0)

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='train_bsdf')
parser.add_argument('--bsdf_type', type=str, default='conductor', help=['some predefined bsdf types, see mit3_bsdf.py'])
parser.add_argument('--isotropic', type=bool, default=True)
parser.add_argument('--wi_size', type=int, default=1024)
parser.add_argument('--wo_size_per_wi', type=int, default=32 * 32)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--mu', type=int, default=32)
args = parser.parse_args()
print(args)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.bsdf_model = MLP(mu=args.mu).to(args.device)
        self.sampling_model = nfm_cond.NFMCond().to(args.device)

        self.dataset = SampleDataset(args)

        self.optim_bsdf = torch.optim.Adam(self.bsdf_model.parameters(),
                                           lr=args.lr, eps=1e-15)
        self.optim_sampling = torch.optim.Adam(self.sampling_model.parameters(), lr=args.lr * 10)

        os.makedirs(os.path.join('log', args.exp_name), exist_ok=True)
        self.log_path = os.path.join('log', args.exp_name)

    def train_network(self):
        with tqdm(total=args.epochs) as pbar:
            for epoch in range(args.epochs):
                self.dataset.shuffle()
                num_batches = int(self.dataset.wi.shape[0] / args.batch_size)
                self.bsdf_model.train()
                self.sampling_model.train()
                # iterate over batches
                for k in range(num_batches):
                    self.optim_bsdf.zero_grad()
                    self.optim_sampling.zero_grad()

                    wi, wo, _, rgb_true = self.dataset.get_traindata(k * self.dataset.bs)
                    loss_sampling = self.sampling_model.forward_kld(wi, wo)

                    rgb_pred = self.bsdf_model(wi, wo).to(args.device)
                    # lower weight for grazing angle
                    # rgb_pred = loss_func.brdf_to_rgb(wi, wo, rgb_pred)  #
                    # rgb_true = loss_func.brdf_to_rgb(wi, wo, rgb_true.reshape(-1, 3))
                    # log transform
                    rgb_true = loss_func.mu_transform(rgb_true, args.mu).reshape(-1, 3)
                    rgb_pred = loss_func.mu_transform(rgb_pred, args.mu)
                    loss_bsdf = F.l1_loss(rgb_pred, rgb_true)

                    loss = loss_bsdf + loss_sampling
                    loss.backward()
                    self.optim_bsdf.step()
                    self.optim_sampling.step()

                if (epoch + 1) % 10 == 0:
                    self.eval(epoch + 1)
                pbar.set_postfix(loss=loss.item(), bsdf_loss=loss_bsdf.item(), sampling_loss=loss_sampling.item())
                pbar.update(1)

            torch.save(self.bsdf_model.state_dict(), os.path.join(self.log_path, 'bsdf_network.pth'))
            torch.save(self.sampling_model.state_dict(), os.path.join(self.log_path, 'sampling_network.pth'))

    @torch.no_grad()
    def eval(self, iter):
        self.bsdf_model.eval()
        self.sampling_model.eval()
        self.vis_pdf(iter)
        self.vis_ndf(iter)

    @torch.no_grad()
    def vis_pdf(self, iter, wi=[0, 0, 1], resolution=512):
        self.sampling_model.eval()
        angles, mask = generate_query_angles(wi, resolution, device=self.args.device)
        # get pdf
        log_pdf = self.sampling_model.log_prob(angles[:, :3], angles[:, 3:])
        pdf = torch.exp(log_pdf).reshape(resolution, resolution)
        pdf = pdf * angles[:, 5].reshape(resolution, resolution)  # disk domain to hemisphere domain
        pdf[mask] = 0.
        os.makedirs(os.path.join(self.log_path, 'pdf'), exist_ok=True)
        mi.util.write_bitmap(os.path.join(self.log_path, 'pdf', f'pdf_{iter}.exr'), pdf.cpu().numpy())

    @torch.no_grad()
    def vis_ndf(self, iter, wi=[0, 0, 1], resolution=512):
        self.bsdf_model.eval()
        angles, mask = generate_query_angles(wi, resolution, device=self.args.device)
        value = self.bsdf_model(angles[:, :3], angles[:, 3:]).reshape(resolution, resolution, -1)
        value[mask] = 0.
        os.makedirs(os.path.join(self.log_path, 'ndf'), exist_ok=True)
        mi.util.write_bitmap(os.path.join(self.log_path, 'ndf', f'ndf_{iter}.exr'), value.cpu().numpy())


if __name__ == '__main__':
    trainer = Trainer(args)
    trainer.train_network()
