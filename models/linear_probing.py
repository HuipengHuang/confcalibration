from .naive_model import NaiveModel
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from scores import utils
import torch.nn as nn

from dataset.utils import merge_dataloader

class LinearProbing(NaiveModel):
    def __init__(self, net, device, args):
        super().__init__(net, device, args)
        out_feature = args.num_classes
        self.T = nn.Parameter(torch.ones(size=(out_feature, out_feature), device=device), requires_grad=False)

    def forward(self, x):
        return self.net(x) @ self.T

    def calibrate(self, cal_loader, test_loader, threshold=None):
        self.net.eval()
        self.T = nn.Parameter(torch.ones(size=(self.args.num_classes, self.args.num_classes), device=self.device), requires_grad=True)
        optimizer = optim.Adam([self.T], self.args.learning_rate)

        if self.args.algorithm == "cp":
            dataloader = merge_dataloader(self.args, cal_loader, test_loader)
            for epoch in range(10):
                for data, target in tqdm(dataloader, desc=f"{epoch} / 100"):
                    data, target = data.to(self.device), target.to(self.device)
                    logits = self.net(data) @ self.T
                    prob = torch.softmax(logits, dim=-1)

                    score = self.score_function(prob)
                    prediction_set = (score <= threshold).to(torch.int)

                    gap = torch.sum(prob * prediction_set) - (1 - self.alpha)

                    loss = gap ** 2
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        else:
            self.T = nn.Parameter(torch.ones(size=(self.net.out_feature, self.net.out_feature)), requires_grad=True)
            optimizer = optim.Adam(self.T, self.args.learning_rate)
            dataloader = cal_loader
            loss_function = nn.CrossEntropyLoss()
            for epoch in range(10):
                for data, target in tqdm(dataloader, desc=f"{epoch} / 100"):
                    data, target = data.to(self.device), target.to(self.device)
                    logits = self.net(data) @ self.T
                    loss = loss_function(logits, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
