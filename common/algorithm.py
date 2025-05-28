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

        cal_loader, tune_loader, test_loader = build_cal_test_loader(args)

        trainer = get_trainer(args)

        if args.epochs:
            train_loader = build_train_dataloader(args)
            trainer.train(train_loader, args.epochs)

        trainer.predictor.calibrate(cal_loader)

        result_dict = trainer.predictor.evaluate(test_loader)

        for key, value in result_dict.items():
            print(f'{key}: {value}')

        if args.save == "True":
            save_exp_result(args, result_dict)

def tune(args):
    bf_covgap = []
    aft_covgap = []
    tuning_bias_list = []

    train_one_model_first(args)

    for run in range(args.num_runs):
        seed = args.seed
        if seed:
            set_seed(seed+run)

        cal_loader, tune_loader, test_loader = build_cal_test_loader(args)

        trainer = get_trainer(args)

        if args.epochs:
            train_loader = build_train_dataloader(args)
            trainer.train(train_loader, args.epochs)

        trainer.predictor.calibrate(cal_loader)
        bf_tune_result_dict = trainer.predictor.evaluate(test_loader)

        trainer.model.tune(tune_loader)

        trainer.predictor.calibrate(cal_loader)
        aft_tune_result_dict = trainer.predictor.evaluate(test_loader)

        for key, value in bf_tune_result_dict.items():
            print(f'After tuning: {key}: {value}')

        tuning_bias = abs(aft_tune_result_dict["Coverage"] - (1 - args.alpha)) -  abs(bf_tune_result_dict["Coverage"] - (1 - args.alpha))
        final_result_dict = {"TuningBias": tuning_bias}

        for key, value in bf_tune_result_dict.items():
            final_result_dict["bf_tune_" + key] = value

        for key, value in aft_tune_result_dict.items():
            final_result_dict["aft_tune_" + key] = value

        bf_covgap.append(abs(bf_tune_result_dict["Coverage"] - (1 - args.alpha)))
        aft_covgap.append(abs(aft_tune_result_dict["Coverage"] - (1 - args.alpha)))
        tuning_bias_list.append(tuning_bias)

        if args.save == "True":
            save_exp_result(args, final_result_dict)

    bf_covgap = np.array(bf_covgap)
    aft_covgap = np.array(aft_covgap)
    tuning_bias_list = np.array(tuning_bias_list)

    mean_bf_covgap = np.mean(bf_covgap)
    mean_aft_covgap = np.mean(aft_covgap)
    mean_tuning_bias = np.mean(tuning_bias_list)

    std_bf_covgap = np.std(bf_covgap)
    std_aft_covgap = np.std(aft_covgap)
    std_tuning_bias = np.std(tuning_bias_list)

    mean_result_dict = {"num_runs":args.num_runs, "mean_bf_covgap": mean_bf_covgap,"mean_aft_covgap":mean_aft_covgap, "mean_tuning_bias":mean_tuning_bias,
                        "std_bf_covgap":std_bf_covgap, "std_aft_covgap":std_aft_covgap, "std_tuning_bias":std_tuning_bias}
    save_exp_result(args, mean_result_dict, path=f"./experiment/{args.algorithm}/mean_result")


def standard(args):
    for run in range(args.num_runs):
        seed = args.seed
        if seed:
            set_seed(seed + run)

        trainer = get_trainer(args)

        if args.epochs:
            train_loader = build_train_dataloader(args)
            trainer.train(train_loader, args.epochs)

        _, _, test_loader = build_cal_test_loader(args)

        result_dict = trainer.predictor.evaluate(test_loader)

        for key, value in result_dict.items():
            print(f'{key}: {value}')

        if args.save == "True":
            save_exp_result(args, result_dict)

def train_one_model_first(args):
    args.load = "False"
    train_loader = build_train_dataloader(args)
    trainer = get_trainer(args)
    trainer.train(train_loader, 1)
    save_model(args, trainer.model)
    args.load = "True"
