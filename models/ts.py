from .naive_model import NaiveModel
import torch.nn as nn
import torch
import torch.optim as optim
class TemperatureScaling(NaiveModel):
    def __init__(self, net, device, args):
        super().__init__(net, device, args)
        self.T = nn.Parameter(torch.tensor([1]).to(self.device), requires_grad=False)

    def forward(self, x):
        return self.net(x) / self.T

    def tune(self, tune_loader):
        self.net.eval()
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