import torchvision.transforms.functional as TF
from wilds import get_dataset
import pandas as pd
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


class rxrx1_dataset(Dataset):
    def __init__(self, train=False):
        super().__init__()
        myTransform = initializeRxrx1Transform()

        ### Load in data from WILDS repository
        rx1Data = get_dataset(dataset="rxrx1", download=True,)
        if train:
            self.rx1Images = rx1Data.get_subset(
                "train",
                transform=myTransform
            )
            metaData = pd.read_csv('data/rxrx1_v1.0/metadata.csv')
            self.metaData = metaData[metaData['dataset'] == 'train']
        else:
            self.rx1Images = rx1Data.get_subset(
                "test",
                transform=myTransform
            )
            metaData = pd.read_csv('data/rxrx1_v1.0/metadata.csv')
            self.metaData = metaData[metaData['dataset'] == 'test']
    def __len__(self):
        return len(self.rx1Images)

    def __getitem__(self, idx):
        return self.rx1Images[idx][0], self.rx1Images[idx][1]

def initializeRxrx1Transform():
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)

    t_standardize = transforms.Lambda(lambda x: standardize(x))

    transforms_ls = [
        transforms.ToTensor(),
        t_standardize,
    ]
    return transforms.Compose(transforms_ls)