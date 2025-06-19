import torch.nn as nn
from scores.utils import get_score

class NaiveModel(nn.Module):
    def __init__(self, net, device, args):
        super().__init__()
        self.net = net
        self.device = device
        self.args = args
        self.score_function = get_score(self.args)
        self.alpha = args.alpha
    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def forward(self, x):
        return self.net(x)

    def calibrate(self, cal_loader, test_loader, threshold):
        raise NotImplementedError

    def get_featurizer(self):
        featurizer = self.net
        classifier = self.net.fc
        featurizer.fc = nn.Identity()
        self.net = nn.Sequential(*(featurizer, classifier))
        return featurizer