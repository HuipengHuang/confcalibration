import torch
from tqdm import tqdm
import models
from loss.utils import get_loss_function
from .utils import get_optimizer
from predictors.get_predictor import get_predictor
class Trainer:
    """
    Trainer class that implement all the functions regarding training.
    All the arguments are passed through args."""
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.utils.build_model(device=self.device, args=args)
        self.batch_size = args.batch_size

        self.optimizer = get_optimizer(args, self.model)

        self.predictor = get_predictor(args, self.model)

        self.num_classes = args.num_classes
        self.loss_function = get_loss_function(args, self.predictor)

    def train_batch(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)
        logits = self.model(data)
        loss = self.loss_function(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self, data_loader, epochs):
        self.model.train()

        for epoch in range(epochs):
            self.adjust_learning_rate(epoch)
            for data, target in tqdm(data_loader, desc=f"Epoch: {epoch} / {epochs}"):
                self.train_batch(data, target)

        if self.args.save_model == "True":
            models.utils.save_model(self.args, self.model)

    def adjust_learning_rate(self, epoch):
        if epoch == 60 or epoch == 120 or epoch == 160:
            self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] / 5

