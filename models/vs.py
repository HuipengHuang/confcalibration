from .naive_model import NaiveModel
import torch.nn as nn
import torch
import torch.optim as optim

class VectorScaling(NaiveModel):
    def __init__(self, net, device, args):
        super().__init__(net, device, args)
        self.freeze_num = args.freeze_num if args.freeze_num else 0

        self.w = nn.Parameter((torch.ones(args.num_classes).to(self.device)), requires_grad=False)
        self.frozen_indices = torch.randperm(args.num_classes)[:self.freeze_num]

        self.b = nn.Parameter((torch.zeros(args.num_classes)).to(self.device), requires_grad=False)

    def forward(self, x):
        return self.net(x) * self.w + self.b


    def tune(self, tune_loader):
        self.net.eval()
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