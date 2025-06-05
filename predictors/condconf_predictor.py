from scores.utils import get_score
import torch
import math
import torch.nn as nn
from loss.pinball_loss import PinballLoss
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from .condconf import setup_cvx_problem_calib
import cvxpy as cp


class CondConfPredictor:
    def __init__(self, args, model):
        self.score_function = get_score(args)
        self.model = model
        self.cal_score = None
        self.alpha = args.alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.cal_prob = None
        self.num_classes = args.num_classes
        self.pinball_loss = PinballLoss(1 - self.alpha)
        self.featurizer = self.model.get_featurizer()

        self.scoresCal = None
        self.y_cal = None
        self.cal_featureMap = None

        self.y_train = None
        self.train_featureMap = None

    def cvForFeatures(self, X, y, numCs=20, minC=0.001, maxC=0.1):
        folds = KFold(n_splits=3, shuffle=True)
        #folds = StratifiedKFold(n_splits=3, shuffle=True)
        Cvalues = np.linspace(minC, maxC, numCs)
        losses = np.zeros(numCs)
        count = 0
        for C in tqdm(Cvalues):
            model = LogisticRegression(C=C, max_iter=5000)
            for i, (trainPoints, testPoints) in enumerate(folds.split(X)):
                reg = model.fit(X[trainPoints, :], y[trainPoints])
                predictedProbs = reg.predict_proba(X[testPoints, :])
                for j in range(len(testPoints)):
                    losses[count] = losses[count] - np.log(predictedProbs[j, int(y[testPoints][j])]) / len(testPoints)
            count = count + 1

        return Cvalues, losses

    def get_train_features(self, train_loader):
        self.model.eval()
        trainfeatureMap = torch.tensor([], device=self.device)
        y_train = torch.tensor([], device=self.device, dtype=torch.int)

        for data, target in tqdm(train_loader, desc="Processing Train Feature"):
            data = data.to(self.device)
            target = target.to(self.device)

            trainfeatureMap = torch.cat((trainfeatureMap, self.featurizer(data)), dim=0)
            y_train = torch.cat((y_train, target), dim=0)


        self.y_train = y_train.detach().cpu().numpy()
        self.train_featureMap = trainfeatureMap.detach().cpu().numpy()

    def computeFeatures(self, XTrain, XCal, XTest, yTrain, Cvalues, losses):
        model = LogisticRegression(C=Cvalues[np.argmin(losses)], max_iter=5000)
        reg = model.fit(XTrain, yTrain)

        featuresCal = reg.predict_proba(XCal)
        featuresTest = reg.predict_proba(XTest)

        return featuresCal, featuresTest

    def computeCoverages(self, XCal, scoresCal, XTest, scoresTest, y_test, alpha):
        coveragesCond = np.zeros(len(XTest))
        set_size = np.zeros(len(XTest))

        scoresCal_numpy = scoresCal[np.arange(self.y_cal.shape[0]), self.y_cal]

        for i in tqdm(range(len(XTest))):
            prob = setup_cvx_problem_calib(1 - alpha, None,
                                           np.concatenate((scoresCal_numpy, np.array([scoresTest[i, y_test[i]]]))),
                                           np.vstack((XCal, XTest[i, :])), {})
            if "MOSEK" in cp.installed_solvers():
                prob.solve(solver="MOSEK")
            else:
                prob.solve()
            threshold = XTest[i, :] @ prob.constraints[2].dual_value
            set_size[i] = np.sum(scoresTest[i] <= threshold)
            coveragesCond[i] = (scoresTest[i, y_test[i]] <= threshold)
        return set_size, coveragesCond


    def calibrate(self, cal_loader, alpha=None):
        self.model.eval()
        calfeatureMap = torch.tensor([], device=self.device)
        y_cal = torch.tensor([], device=self.device, dtype=torch.int)
        scoresCal = torch.tensor([], device=self.device)

        for data, target in cal_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            logits = self.model(data)
            prob = torch.softmax(logits, dim=-1)
            batch_score = self.score_function(prob)

            calfeatureMap = torch.cat((calfeatureMap, self.featurizer(data)), dim=0)
            y_cal = torch.cat((y_cal, target), dim=0)
            scoresCal = torch.cat((scoresCal, batch_score), 0)

        self.scoresCal = scoresCal.detach().cpu().numpy()
        self.y_cal = y_cal.detach().cpu().numpy()
        self.cal_featureMap = calfeatureMap.detach().cpu().numpy()

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

        testfeatureMap = torch.tensor([], device=self.device)
        test_target = torch.tensor([], device=self.device, dtype=torch.int)

        scoresTest = torch.tensor([], device=self.device)
        for data, target in test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            logits = self.model(data)
            prob = torch.softmax(logits, dim=-1)
            batch_score = self.score_function(prob)

            testfeatureMap = torch.cat((testfeatureMap, self.featurizer(data)), dim=0)
            test_target = torch.cat((test_target, target), dim=0)
            scoresTest = torch.cat((scoresTest, batch_score), 0)

        testfeatureMap = testfeatureMap.detach().cpu().numpy()
        test_target = test_target.detach().cpu().numpy()
        scoresTest = scoresTest.detach().cpu().numpy()

        Cvalues, losses = self.cvForFeatures(self.train_featureMap, self.y_train,
                                             numCs=20, minC=0.001, maxC=0.1)

        finalFeaturesCal, finalFeaturesTest = self.computeFeatures(self.train_featureMap,
                                                                   self.cal_featureMap,
                                                                   testfeatureMap,
                                                                   self.y_train, Cvalues, losses)

        set_size, coveragesCond = self.computeCoverages(finalFeaturesCal, self.scoresCal,
                                                         finalFeaturesTest, scoresTest, test_target, alpha=0.1)

        result_dict = {"Coverage":np.mean(coveragesCond),
                       "AverageSetSize":np.mean(set_size),
                       }

        return result_dict

