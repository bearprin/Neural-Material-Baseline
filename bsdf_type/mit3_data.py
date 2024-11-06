import mitsuba as mi
import drjit as dr
import torch

mi.set_variant('cuda_ad_rgb')

from utils import generate_query_angles

# material type
diffuse_dict = {'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.2, 0.25, 0.7]
                }
                }
conductor = {'type': 'roughconductor',
             'material': 'Al',
             'distribution': 'ggx',
             'alpha': 0.1
             }
aniso_conductor = {'type': 'roughconductor',
                   'material': 'Al',
                   'distribution': 'ggx',
                   'alpha_u': 0.05,
                   'alpha_v': 0.3
                   }
plastic = {'type': 'roughplastic',
           'distribution': 'ggx',
           'int_ior': 1.61,
           'diffuse_reflectance': {
               'type': 'rgb',
               'value': 0
           }
           }
measured_dict = {'type': 'measured',
                 'filename': 'cc_nothern_aurora_spec.bsdf'
                 }

disney = {'type': 'principled',
          'base_color': {
              'type': 'rgb',
              'value': [1.0, 1.0, 1.0]
          },
          'metallic': 0.7,
          'specular': 0.6,
          'roughness': 0.2,
          'spec_tint': 0.4,
          'anisotropic': 1.0,
          'sheen': 0.3,
          'sheen_tint': 0.2,
          'clearcoat': 0.6,
          'clearcoat_gloss': 0.3,
          'spec_trans': 0.4
          }
hair = {
    'type': 'hair',
    'eumelanin': 0.2,
    'pheomelanin': 0.4
}


class Mit3BSDF:
    def __init__(self, args):
        bsdf_type = args.bsdf_type
        isotropic = args.isotropic
        self.bsdf_type = bsdf_type
        self.isotropic = isotropic
        self.args = args

        if bsdf_type == 'diffuse':
            self.bsdf = mi.load_dict(diffuse_dict)
        elif bsdf_type == 'conductor':
            self.bsdf = mi.load_dict(conductor) if isotropic else mi.load_dict(aniso_conductor)
        elif bsdf_type == 'measured':
            self.bsdf = mi.load_dict(measured_dict)
        elif bsdf_type == 'plastic':
            self.bsdf = mi.load_dict(plastic)
        elif bsdf_type == 'disney':
            self.bsdf = mi.load_dict(disney)
        elif bsdf_type == 'hair':
            self.bsdf = mi.load_dict(hair)

        # self.ndf_img, self.pdf_img = self._ndf_pdf_img()
        self.wi_size = args.wi_size
        self.wo_size_per_wi = args.wo_size_per_wi

    def _ndf_pdf_img(self, wi=[0, 0, 1], resolution=512):
        angles, mask = generate_query_angles(wi, resolution, 'cpu')

        wi = angles[:, :3]
        wo = angles[:, 3:]
        ctx = mi.BSDFContext()
        wi = mi.Vector3f(wi.numpy())
        wo = mi.Vector3f(wo.numpy())
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = wi

        pdf = self.bsdf.pdf(ctx, si, wo, True).numpy()
        pdf = pdf.reshape(resolution, resolution)
        pdf[mask] = 0.
        mi.util.write_bitmap(f"pdf_gt_{self.bsdf_type}.exr", pdf)

        eval_value = self.bsdf.eval(ctx, si, wo, True).numpy()
        eval_value = eval_value.reshape(resolution, resolution, -1)
        eval_value[mask] = 0.
        mi.util.write_bitmap(f"ndf_gt_{self.bsdf_type}.exr", eval_value)

        return eval_value, pdf

    def generate_pdf_data(self):
        self.wi_sampler = mi.load_dict({'type': 'stratified', 'sample_count': 1})
        self.wi_sampler.seed(42, self.wi_size)
        self.wo_sampler = mi.load_dict({'type': 'stratified', 'sample_count': self.wo_size_per_wi})
        self.wo_sampler.set_sample_count(self.wo_size_per_wi)
        self.wo_sampler.set_samples_per_wavefront(self.wo_size_per_wi)
        self.wo_sampler.seed(42, self.wi_size * self.wo_size_per_wi)

        wi = mi.warp.square_to_uniform_hemisphere(self.wi_sampler.next_2d())
        ori_wi = wi
        # 1. repeat
        wi = dr.repeat(wi, self.wo_size_per_wi)
        # 2. tile
        # wi = wi.torch()
        # wi = torch.tile(wi, (self.wo_size_per_wi, 1))

        ctx = mi.BSDFContext()
        wi = mi.Vector3f(wi)
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = wi

        bs, throughput = self.bsdf.sample(ctx, si, self.wo_sampler.next_1d(), self.wo_sampler.next_2d(), True)
        eval_value = self.eval(torch.cat([wi.torch(), bs.wo.torch()], dim=-1))
        wi = ori_wi.torch()
        wo = bs.wo.torch()
        pdf = bs.pdf.torch()
        # 1. repeat
        wo = wo.reshape(self.wi_size, self.wo_size_per_wi, 3)
        pdf = pdf.reshape(self.wi_size, self.wo_size_per_wi)
        eval_value = eval_value.reshape(self.wi_size, self.wo_size_per_wi, 3)
        # 2. tile
        # wo = torch.stack(torch.chunk(wo, self.wo_size_per_wi, dim=0), dim=1)
        # pdf = torch.stack(torch.chunk(pdf, self.wo_size_per_wi, dim=0), dim=1)
        # check the first wo for each wi
        # wi_ = mi.Vector3f(wi)
        # wo_ = mi.Vector3f(wo[:, 0])
        # si.wi = wi_
        # pdf_ = pdf[:, 0]
        # value_ = eval_value[:, 0]
        # pdf_gt = self.bsdf.pdf(ctx, si, wo_, True).torch()
        # value_gt = self.bsdf.eval(ctx, si, wo_, True).torch()
        # print('PDF Check error', (pdf_gt - pdf_))
        # print('Value Check error', (value_gt - value_))
        return wi, wo, pdf, eval_value

    def eval_pdf(self, io_rvectors):
        wi, wo = io_rvectors[:, :3], io_rvectors[:, 3:]

        ctx = mi.BSDFContext()
        wi = mi.Vector3f(wi)
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = wi
        wo = mi.Vector3f(wo)

        value, pdf = self.bsdf.eval_pdf(ctx, si, wo, True)  # (N, 3)

        cos_theta_o = mi.Frame3f.cos_theta(wo)
        value = value / dr.maximum(cos_theta_o, 1e-12)
        value = value.torch()
        pdf = pdf.torch()
        return value, pdf

    def eval(self, io_rvectors):
        wi, wo = io_rvectors[:, :3], io_rvectors[:, 3:]

        ctx = mi.BSDFContext()
        wi = mi.Vector3f(wi)
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = wi
        wo = mi.Vector3f(wo)

        res = self.bsdf.eval(ctx, si, wo, True)  # (N, 3)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        res = res / dr.maximum(cos_theta_o, 1e-12)
        res = res.torch()
        return res

    def pdf(self, io_rvectors):
        wi, wo = io_rvectors[:, :3], io_rvectors[:, 3:]
        ctx = mi.BSDFContext()
        wi = mi.Vector3f(wi)
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = wi
        wo = mi.Vector3f(wo)

        pdf = self.bsdf.pdf(ctx, si, wo, True)
        pdf = pdf.torch()
        return pdf


if __name__ == '__main__':
    sampler = Mit3BSDF('disney')
    sampler.generate_pdf_data()
