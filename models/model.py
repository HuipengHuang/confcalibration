import math

import torch.nn as nn

import torch
import torch.optim as optim
import torch.nn.functional as F
from common.utils import soft_quantile

class BaseModel(nn.Module):
    def __init__(self, net, device, args):
        super(BaseModel, self).__init__()
        self.net = net
        self.device = device
        self.args = args
    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def forward(self, x):
        raise self.net(x)

    def tune(self, tune_loader):
        raise NotImplementedError


class TemperatureScaling(BaseModel):
    def __init__(self, net, device, args):
        super().__init__(net, device, args)
        self.T = nn.Parameter(torch.tensor([1]).to(self.device), requires_grad=False)

    def forward(self, x):
        return self.net(x) / self.T

    def tune(self, tune_loader):
        self.T = nn.Parameter(torch.tensor([1.5]).to(self.device), requires_grad=True)
        optimizer = optim.LBFGS([self.T], lr=0.1, max_iter=200)

        logits_list= []
        labels_list = []

        with torch.no_grad():
            for data, target in tune_loader:
                data, target= data.to(self.device), target.to(self.device)
                logits = self.net(data)
                logits_list.append(logits)
                labels_list.append(target)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        Criterion = nn.CrossEntropyLoss()
        def compute_loss():
            optimizer.zero_grad()
            out = logits / self.T
            loss = Criterion(out, labels)
            loss.backward()
            return loss
        optimizer.step(compute_loss)


class PlattScaling(BaseModel):
    def __init__(self, net, device, args):
        super().__init__(net, device, args)
        self.a = nn.Parameter(torch.tensor([1.]).to(device), requires_grad=False)
        self.b = nn.Parameter(torch.tensor([0.]).to(device), requires_grad=False)

    def forward(self, x):
        return self.net(x) * self.a + self.b


    def tune(self, tune_loader):
        self.a = nn.Parameter(torch.tensor([1.5]).to(self.device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([1.5]).to(self.device), requires_grad=True)

        optimizer = optim.LBFGS([self.a, self.b], lr=0.1, max_iter=100)

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for data, target in tune_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.net(data)
                logits_list.append(logits)
                labels_list.append(target)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        Criterion = nn.CrossEntropyLoss()

        def eval():
            optimizer.zero_grad()
            out = logits * self.a + self.b
            loss = Criterion(out, labels)
            loss.backward()
            return loss

        optimizer.step(eval)



class VectorScaling(BaseModel):
    def __init__(self, net, device, args):
        super().__init__(net, device, args)
        self.freeze_num = args.freeze_num if args.freeze_num else 0

        self.w = nn.Parameter((torch.ones(args.num_classes).to(self.device)), requires_grad=False)
        self.frozen_indices = torch.randperm(args.num_classes)[:self.freeze_num]

        self.b = nn.Parameter((torch.zeros(args.num_classes)).to(self.device), requires_grad=False)

    def forward(self, x):
        return self.net(x) * self.w + self.b


    def tune(self, tune_loader):
        self.w = nn.Parameter((torch.ones(self.args.num_classes) * 1.5).to(self.device), requires_grad=True)

        self.b = nn.Parameter((torch.rand(self.args.num_classes) * 2.0 - 1.0).to(self.device), requires_grad=True)

        optimizer = optim.LBFGS([self.w, self.b], lr=0.1, max_iter=300)

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for data, target in tune_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.net(data)
                logits_list.append(logits)
                labels_list.append(target)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        Criterion = nn.CrossEntropyLoss()

        def compute_loss():
            optimizer.zero_grad()
            out = logits * self.w + self.b
            loss = Criterion(out, labels)
            loss.backward()
            self.w.grad[self.frozen_indices] = 0
            return loss

        optimizer.step(compute_loss)



class ConfTr(BaseModel):
    def __init__(self, net, device, args):
        super().__init__(net, device, args)
        self.num_epochs = 50
        self.projection_weight = nn.Parameter(torch.eye(args.num_classes, device=device), requires_grad=False)
        self.projection_bias = nn.Parameter(torch.zeros(args.num_classes, device=device), requires_grad=False)
        self.alpha = 0.01


    def forward(self, logits):
        return torch.matmul(logits, self.projection_weight.T) + self.projection_bias

    def tune(self, tune_loader):
        self.net.train()
        self.projection_weight = nn.Parameter(torch.eye(self.args.num_classes, device=self.device), requires_grad=True)
        self.projection_bias = nn.Parameter(torch.zeros(self.args.num_classes, device=self.device), requires_grad=True)
        Criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam([self.projection_weight, self.projection_bias], lr=0.001)

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            for input, label in tune_loader:
                input, label = input.to(self.device), label.to(self.device)
                logits = self.net(input)
                out = torch.matmul(logits, self.projection_weight.T) + self.projection_bias
                val_tau = self.smooth_calibrate_fn(out, label)
                loss = 0.00001 * self.smooth_predict_fn(out, val_tau).mean() + Criterion(out, label)
                loss.backward()
            optimizer.step()


    def _q(self, n, alpha):
        if alpha is None:
            alpha = self.alpha
        q = math.ceil((n + 1) * (1 - alpha)) / n
        if q > 1:
            return 1.0
        else:
            return q

    def smooth_calibrate_fn(self, logits, labels):
        n = logits.shape[0]
        log_probabilities = F.softmax(logits, dim=1)
        conformity_scores = log_probabilities[
            torch.arange(log_probabilities.shape[0]), labels]
        threshold = soft_quantile(-conformity_scores, self._q(n, self.alpha))
        return threshold

    def smooth_predict_fn(self, logits, tau):
        log_probabilities = F.softmax(logits, dim=1)
        membership_logits = tau - (-log_probabilities)
        membership_scores = torch.sigmoid(membership_logits / 0.1)
        return torch.clamp(membership_scores.sum(-1) - 1, min=0) + 1