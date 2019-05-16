import torch.utils.data as data


class COCOStuff164K(data.Dataset):
    def __init__(self, root):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
