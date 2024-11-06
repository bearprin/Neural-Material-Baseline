

import time


import mitsuba as mi
import drjit as dr
from tqdm import tqdm

mi.set_variant("cuda_ad_rgb")

import torch
torch.set_grad_enabled(False)

from model.MLP import MLP
from model.nfm_cond import NFMCond

dr.set_flag(dr.JitFlag.VCallRecord, False)
# dr.set_flag(dr.JitFlag.LoopRecord, False)



class NetworkBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        # Set the BSDF flags
        reflection_flags = (
                mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
        )
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags

        # load eval pdf
        self.eval_network = MLP().to('cuda')
        self.eval_network.load_state_dict(torch.load(r'log/train_bsdf/bsdf_network.pth'))
        self.eval_network.eval()

        # load sampling network
        self.sampling_network = NFMCond().to('cuda')
        self.sampling_network.load_state_dict(torch.load(r'log/train_bsdf/sampling_network.pth'))
        self.sampling_network.eval()

    def sample(self, ctx, si, sample1, sample2, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active &= cos_theta_i > 0
        bs = mi.BSDFSample3f()
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.wo, bs.pdf = self._sample_network(si.wi)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.sampled_component = 0

        # proxy
        # bs, gt_v = self.proxy_brdf.sample(ctx, si, sample1, sample2, active)
        # wi = si.wi.torch()
        # wo, pdf = self.sapmpling_network.sample(wi, 1)
        # bs.wo = mi.Vector3f(wo[..., 0], wo[..., 1], wo[..., 2])
        # bs.pdf = mi.Float(pdf)
        cos_theta_o = mi.Frame3f.cos_theta(bs.wo)
        value_mi = self._eval_network(si.wi, bs.wo)
        value_mi = value_mi / bs.pdf

        return (bs, dr.select(active & (bs.pdf > 0.0), value_mi, mi.Vector3f(0)))

    def _eval_network(self, wi, wo):
        wi = wi.torch()
        wo = wo.torch()
        value = self.eval_network(wi, wo)
        value_mi = mi.Vector3f(value[..., 0], value[..., 1], value[..., 2])
        return value_mi

    def _sample_network(self, wi):
        wi = wi.torch()
        wo, log_pdf = self.sampling_network.sample(wi)
        pdf = torch.exp(log_pdf)
        z = torch.sqrt(1 - wo[..., 0] ** 2 +  wo[..., 1] ** 2)
        pdf = pdf * z
        pdf[torch.isinf(pdf) | torch.isnan(pdf)] = 0.0
        wo = mi.Vector3f(wo[..., 0], wo[..., 1], z)
        pdf = mi.Float(pdf)
        return wo, pdf

    def _pdf_network(self, wi, wo):
        wi = wi.torch()
        wo = wo.torch()
        log_pdf = self.sampling_network.log_prob(wi, wo)
        pdf = torch.exp(log_pdf)
        z = torch.sqrt(1 - wo[..., 0] ** 2 +  wo[..., 1] ** 2)
        pdf = pdf * z
        pdf[torch.isinf(pdf) | torch.isnan(pdf)] = 0.0
        pdf = mi.Float(pdf)
        return pdf

    def eval(self, ctx, si, wo, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active &= (cos_theta_i > 0.0) & (cos_theta_o > 0.0)

        value_mi = self._eval_network(si.wi, wo)
        value_mi = value_mi * cos_theta_o

        return dr.select(
            active, value_mi, mi.Vector3f(0)
        )

    def pdf(self, ctx, si, wo, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        pdf = self._pdf_network(si.wi, wo)
        # pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def traverse(self, callback):
        return

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        return "MyBSDF[\n" "    albedo=%s,\n" "]" % (self.albedo)


if __name__ == "__main__":
    mi.register_bsdf("network_bsdf", lambda props: NetworkBSDF(props))

    scene = mi.load_file("./sphere_scene/scene.xml")
    params = mi.traverse(scene)
    print(params)
    # batch rendering for large spp
    with dr.suspend_grad():
        spp = 2
        iters = 128
        start_time = time.time()
        image = mi.render(scene, spp=spp).numpy()
        for i in tqdm(range(1, iters)):
            image += mi.render(scene, spp=spp, seed=i).numpy()
        image /= iters
        end_time = time.time()
        mi.util.write_bitmap("assets/network_res.png", image)
    print("Render time: " + str(end_time - start_time) + " seconds")
