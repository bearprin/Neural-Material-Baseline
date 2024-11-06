import torch
from torch.utils.data import Dataset

from bsdf_type.mit3_data import Mit3BSDF

class SampleDataset(Dataset):
    def __init__(self, args):
        super(SampleDataset, self).__init__()
        self.bs = args.batch_size
        self.bsdf = Mit3BSDF(args)
        # wi [N, 3], wo [N, M, 3], pdf [N, M], value [N, M, 3]
        self.wi, self.wo, self.pdf, self.value = self.bsdf.generate_pdf_data()



    def __len__(self):
        return self.wi.shape[0]

    def get_traindata(self, idx):
        return self.wi[idx:idx + self.bs, :], self.wo[idx:idx + self.bs], self.pdf[idx:idx + self.bs], self.value[idx:idx + self.bs]

    def shuffle(self):
        r = torch.randperm(self.wi.shape[0])
        self.wi = self.wi[r]
        self.wo = self.wo[r]
        self.pdf = self.pdf[r]
        self.value = self.value[r]

        r = torch.randperm(self.wo.shape[1])
        self.wo = self.wo[:, r]
        self.pdf = self.pdf[:, r]
        self.value = self.value[:, r]


