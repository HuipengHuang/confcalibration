from .naive_model import NaiveModel
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from common.utils import soft_quantile

class ConfTr(NaiveModel):
    def __init__(self, net, device, args):
        super().__init__(net, device, args)
        self.num_epochs = 50
        self.projection_weight = nn.Parameter(torch.eye(args.num_classes, device=device), requires_grad=False)
        self.projection_bias = nn.Parameter(torch.zeros(args.num_classes, device=device), requires_grad=False)
        self.alpha = 0.01


    def forward(self, x):
        logits = self.net(x)
        return torch.matmul(logits, self.projection_weight.T) + self.projection_bias

    def calibrate(self, tune_loader):
        self.net.eval()
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