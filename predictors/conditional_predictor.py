from scores.utils import get_score
import torch
import math
import torch.nn as nn
from loss.pinball_loss import PinballLoss

class ConditionalPredictor:
    def __init__(self, args, model):
        self.score_function = get_score(args)
        self.model = model
        self.cal_score = None
        self.alpha = args.alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.cal_prob = None
        self.num_classes = args.num_classes
        self.pinball_loss = PinballLoss(self.alpha)
        self.weight = None

    def calibrate(self, cal_loader, alpha=None):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""
        self.model.eval()
        with torch.no_grad():
            cal_score = torch.tensor([], device=self.device)
            all_prob = torch.tensor([], device=self.device)
            for data, target in cal_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.model(data)

                prob = torch.softmax(logits, dim=1)
                batch_score = self.score_function.compute_target_score(prob, target)
                all_prob = torch.cat((all_prob, prob), dim=0)
                cal_score = torch.cat((cal_score, batch_score), 0)
            self.cal_score = cal_score
            self.cal_prob = all_prob
        if self.args.split == "True":
            weight = nn.Parameter(torch.zeros(self.num_classes, device=self.device))
            optimizer = torch.optim.Adam([weight], lr=1e-2)
            prev_loss = 0
            diff = 10
            while diff > 1e-1:
                g_x = self.cal_prob @ weight
                loss = self.pinball_loss(g_x, self.cal_score)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                diff = abs(loss - prev_loss)
                prev_loss = loss
            self.weight = weight

    def calibrate_batch_logit(self, logits, target, alpha):
        """Design for conformal training, which needs to compute threshold in every batch"""
        prob = torch.softmax(logits, dim=-1)
        batch_score = self.score_function.compute_target_score(prob, target)
        N = target.shape[0]
        return torch.quantile(batch_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)

    def get_prediction_set(self, test_data_prob):
        if self.args.split == "False":
            test_data_prob = test_data_prob.detach()
            target_score = self.score_function(test_data_prob)
            data_prob = torch.cat((self.cal_prob, test_data_prob.view(1, -1)), dim=0)
            pred_set = torch.zeros(self.num_classes, device=self.device)
            for y in range(self.num_classes):
                weight = nn.Parameter(torch.zeros(self.num_classes, device=self.device))
                optimizer = torch.optim.Adam([weight], lr=1e-2)
                score = torch.cat((self.cal_score, target_score[y].view(1)), dim=0)

                prev_loss = 0
                diff = 10
                while diff > 1e-1:
                    g_x = data_prob @ weight
                    loss = self.pinball_loss(g_x, score)
                    diff = abs(loss.item() - prev_loss)
                    prev_loss = loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if target_score[y] <= test_data_prob @ weight:
                    pred_set[y] = 1
            return torch.tensor(pred_set, device=self.device)
        else:
            threshold = test_data_prob * self.weight
            target_score = self.score_function(test_data_prob)
            return (target_score <= threshold).to(torch.int)


    def evaluate(self, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""
        self.model.eval()
        total_accuracy = 0
        total_coverage = 0
        total_prediction_set_size = 0
        total_samples = 0
        cond_covgap = 0

        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            batch_size = target.shape[0]
            total_samples += batch_size

            logit = self.model(data)
            prob = torch.softmax(logit, dim=-1)
            prediction = torch.argmax(prob, dim=-1)
            total_accuracy += (prediction == target).sum().item()

            for i in range(batch_size):
                pred_set = self.get_prediction_set(prob[i])

                total_prediction_set_size += torch.sum(pred_set)
                if pred_set[target] == 1:
                    total_coverage += 1
                weight = torch.rand(self.num_classes, device=self.device)
                f_x = prob[i] @ weight
                sub_covgap = self.alpha if target in pred_set else self.alpha - 1
                cond_covgap += f_x * sub_covgap

        accuracy = total_accuracy / total_samples
        coverage = total_coverage / total_samples
        avg_set_size = total_prediction_set_size / total_samples
        E_cond_covgap = cond_covgap / total_samples

        result_dict = {
            f"Top1Accuracy": accuracy,
            f"AverageSetSize": avg_set_size,
            f"Coverage": coverage,
            f"ConditionalCoverageGap": E_cond_covgap
        }

        return result_dict

