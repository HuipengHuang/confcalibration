import numpy as np
from models.utils import save_model
from dataset.utils import build_cal_test_loader, build_train_dataloader
from trainers.get_trainer import get_trainer
from common.utils import save_exp_result, set_seed


def cp(args):
    for run in range(args.num_runs):
        seed = args.seed
        if seed:
            set_seed(seed+run)

        cal_loader, _, test_loader = build_cal_test_loader(args)

        trainer = get_trainer(args)


        train_loader = build_train_dataloader(args)
        if args.epochs:
            trainer.train(train_loader, args.epochs)

        if args.cc == "True":
            trainer.predictor.calibrate(cal_loader)
            threshold = trainer.predictor.threshold
        else:
            threshold=None

        trainer.model.calibrate(cal_loader, test_loader, threshold)

        result_dict = trainer.predictor.evaluate(test_loader)

        for key, value in result_dict.items():
            print(f'{key}: {value}')

        if args.save == "True":
            save_exp_result(args, result_dict)


def standard(args):
    for run in range(args.num_runs):
        seed = args.seed
        if seed:
            set_seed(seed + run)

        cal_loader, _, test_loader = build_cal_test_loader(args)

        trainer = get_trainer(args)

        train_loader = build_train_dataloader(args)
        if args.epochs:
            trainer.train(train_loader, args.epochs)

        result_dict = trainer.predictor.evaluate(test_loader)

        for key, value in result_dict.items():
            print(f'{key}: {value}')

        if args.save == "True":
            save_exp_result(args, result_dict)

def train_one_model_first(args):
    args.load = "False"
    train_loader = build_train_dataloader(args)
    trainer = get_trainer(args)
    trainer.train(train_loader, 200)
    save_model(args, trainer.model)
    args.load = "True"
