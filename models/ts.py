from .naive_model import NaiveModel
import torch.nn as nn
import torch
import torch.optim as optim
from dataset.utils import merge_dataloader
from tqdm import tqdm

class TemperatureScaling(NaiveModel):
    def __init__(self, net, device, args):
        super().__init__(net, device, args)
        self.T = nn.Parameter(torch.tensor([1.0]).to(self.device), requires_grad=False)

    def forward(self, x):
        return self.net(x) / self.T

    def calibrate(self, cal_loader, test_loader, threshold=None):
        self.net.eval()

        if self.args.cc == "True":
            self.T = nn.Parameter(torch.tensor([1.5]).to(self.device), requires_grad=True)
            optimizer = optim.LBFGS([self.T], lr=0.1, max_iter=500)

            dataloader = merge_dataloader(self.args, cal_loader, test_loader)

            logits_list = []
            labels_list = []
            with torch.no_grad():
                for data, target in dataloader:
                        data, target = data.to(self.device), target.to(self.device)
                        logits = self.net(data)
                        logits_list.append(logits)
                        labels_list.append(target)
                logits = torch.cat(logits_list).to(self.device)
                labels = torch.cat(labels_list).to(self.device)

                prob = torch.softmax(logits, dim=-1)
                score = self.score_function(prob)

                conf_mask = score <= threshold
            def compute_loss():
                optimizer.zero_grad()
                out = logits / self.T
                prob = torch.softmax(out, dim=-1)
                loss = torch.log((torch.sum(prob * conf_mask) - (1 - self.alpha) * prob.shape[0])**2)
                loss.backward()
                return loss

            optimizer.step(compute_loss)
        else:
            self.T = nn.Parameter(torch.tensor([1.5]).to(self.device), requires_grad=True)
            optimizer = optim.LBFGS([self.T], lr=0.1, max_iter=500)
            Criterion = nn.CrossEntropyLoss()

            logits_list= []
            labels_list = []
            with torch.no_grad():
                for data, target in cal_loader:
                        data, target= data.to(self.device), target.to(self.device)
                        logits = self.net(data)
                        logits_list.append(logits)
                        labels_list.append(target)
                logits = torch.cat(logits_list).to(self.device)
                labels = torch.cat(labels_list).to(self.device)

            def compute_loss():
                optimizer.zero_grad()
                out = logits / self.T
                loss = Criterion(out, labels)
                loss.backward()
                return loss
            optimizer.step(compute_loss)

