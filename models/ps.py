from .naive_model import NaiveModel
import torch.nn as nn
import torch
import torch.optim as optim


class PlattScaling(NaiveModel):
    def __init__(self, net, device, args):
        super().__init__(net, device, args)
        self.a = nn.Parameter(torch.tensor([1.]).to(device), requires_grad=False)
        self.b = nn.Parameter(torch.tensor([0.]).to(device), requires_grad=False)

    def forward(self, x):
        return self.net(x) * self.a + self.b


    def calibrate(self, tune_loader):
        self.net.eval()
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