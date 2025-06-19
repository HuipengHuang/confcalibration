from scores.utils import get_score
import torch
import math
from predictors.utils import compute_calibration_metrics
import torch.nn as nn
from dataset.utils import merge_dataloader


class Predictor:
    def __init__(self, args, model):
        self.score_function = get_score(args)
        self.model = model
        self.threshold = None
        self.alpha = args.alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args

    def calibrate(self, cal_loader, test_loader, alpha=None):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""
        self.model.eval()
        with torch.no_grad():
            if alpha is None:
                alpha = self.alpha
            cal_score = torch.tensor([], device=self.device)
            for data, target in cal_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.model(data)

                prob = torch.softmax(logits, dim=1)
                batch_score = self.score_function.compute_target_score(prob, target)

                cal_score = torch.cat((cal_score, batch_score), 0)
            N = cal_score.shape[0]
            threshold = torch.quantile(cal_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)
            self.threshold = threshold


    def calibrate_batch_logit(self, logits, target, alpha):
        """Design for conformal training, which needs to compute threshold in every batch"""
        prob = torch.softmax(logits, dim=-1)
        batch_score = self.score_function.compute_target_score(prob, target)
        N = target.shape[0]
        return torch.quantile(batch_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)

    def evaluate(self, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""
        self.model.eval()
        with torch.no_grad():
            prob_tensor, label_tensor = torch.tensor([], device=self.device), torch.tensor([], device=self.device)
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                logit = self.model(data)
                prob = torch.softmax(logit, dim=-1)
                prob_tensor = torch.cat((prob_tensor, prob), 0)
                label_tensor = torch.cat((label_tensor, target), 0)
            accuracy, ece, ace, mce, piece = compute_calibration_metrics(prob_tensor, label_tensor)
            result_dict = {
                f"Top1Accuracy": accuracy,
                f"ECE": ece,
                f"ACE": ace,
                f"MCE": mce,
                f"Piece": piece
            }
        return result_dict

"""            with torch.no_grad():
                total_accuracy = 0
                total_coverage = 0
                total_prediction_set_size = 0
                total_samples = 0

                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    batch_size = target.shape[0]
                    total_samples += batch_size

                    logit = self.model(data)
                    prob = torch.softmax(logit, dim=-1)
                    prediction = torch.argmax(prob, dim=-1)
                    total_accuracy += (prediction == target).sum().item()

                    batch_score = self.score_function(prob)
                    prediction_set = (batch_score <= self.threshold).to(torch.int)

                    target_prediction_set = prediction_set[torch.arange(batch_size), target]
                    total_coverage += target_prediction_set.sum().item()

                    total_prediction_set_size += prediction_set.sum().item()


                accuracy = total_accuracy / total_samples
                coverage = total_coverage / total_samples
                avg_set_size = total_prediction_set_size / total_samples
                result_dict = {
                    f"Top1Accuracy": accuracy,
                    f"AverageSetSize": avg_set_size,
                    f"Coverage": coverage,
                }"""