import torch.nn as nn


class NaiveModel(nn.Module):
    def __init__(self, net, device, args):
        super().__init__()
        self.net = net
        self.device = device
        self.args = args
    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def forward(self, x):
        return self.net(x)

    def tune(self, tune_loader):
        raise NotImplementedError

    def get_featurizer(self):
        featurizer = self.net
        classifier = self.net.fc
        featurizer.fc = nn.Identity()
        self.net = nn.Sequential(*(featurizer, classifier))
        return featurizer