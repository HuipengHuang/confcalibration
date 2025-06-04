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


        train_loader = build_train_dataloader(args)
        if args.epochs:
            trainer.train(train_loader, args.epochs)

        if args.predictor == "condconf":
            trainer.predictor.get_train_features(train_loader)
            print("Train_features are processed")

        trainer.predictor.calibrate(cal_loader)

        result_dict = trainer.predictor.evaluate(test_loader)

        for key, value in result_dict.items():
            print(f'{key}: {value}')

        if args.save == "True":
            save_exp_result(args, result_dict)

def tune(args):
    holdout_covgap = []
    tune_covgap = []
    tuning_bias_list = []
    holdout_coverage = []
    tune_coverage = []


    if args.train_one_model_first == "True":
        train_one_model_first(args)

    for run in range(args.num_runs):
        seed = args.seed
        if seed:
            set_seed(seed+run)

        cal_loader, cal_tune_loader, test_loader = build_cal_test_loader(args)

        trainer = get_trainer(args)

        train_loader = build_train_dataloader(args)

        if args.epochs:
            trainer.train(train_loader, args.epochs)

        if args.predictor == "condconf":
            trainer.predictor.get_train_features(train_loader)
            print("Train_features are processed")

        trainer.model.tune(cal_tune_loader)

        trainer.predictor.calibrate(cal_loader)
        holdout_result_dict = trainer.predictor.evaluate(test_loader)

        trainer.predictor.calibrate(cal_tune_loader)
        tune_result_dict = trainer.predictor.evaluate(test_loader)


        tuning_bias = abs(tune_result_dict["Coverage"] - (1 - args.alpha)) -  abs(holdout_result_dict["Coverage"] - (1 - args.alpha))
        final_result_dict = {"TuningBias": tuning_bias}

        print("Using holdout calibration set: ")
        for key, value in holdout_result_dict.items():
            print(f'{key}: {value}')
        print("Using same set for calibration and tune: ")
        for key, value in tune_result_dict.items():
            print(f'{key}: {value}')
        print("Tuning Bias: ", tuning_bias)
        print()
        for key, value in holdout_result_dict.items():
            final_result_dict["holdout_" + key] = value

        for key, value in tune_result_dict.items():
            final_result_dict["tune_" + key] = value

        holdout_covgap.append(abs(holdout_result_dict["Coverage"] - (1 - args.alpha)))
        tune_covgap.append(abs(tune_result_dict["Coverage"] - (1 - args.alpha)))
        holdout_coverage.append(holdout_result_dict["Coverage"])
        tune_coverage.append(tune_result_dict["Coverage"])
        tuning_bias_list.append(tuning_bias)


    holdout_covgap = np.array(holdout_covgap)
    tune_covgap = np.array(tune_covgap)
    tuning_bias_list = np.array(tuning_bias_list)

    mean_holdout_covgap = np.mean(holdout_covgap)
    mean_tune_covgap = np.mean(tune_covgap)
    mean_tuning_bias = np.mean(tuning_bias_list)
    mean_holdout_coverage = np.mean(holdout_coverage)
    mean_tune_coverage = np.mean(tune_coverage)

    std_holdout_covgap = np.std(holdout_covgap)
    std_tune_covgap = np.std(tune_covgap)
    std_tuning_bias = np.std(tuning_bias_list)
    std_holdout_coverage = np.std(holdout_coverage)
    std_tune_coverage = np.std(tune_coverage)

    mean_result_dict = {"num_runs":args.num_runs, "mean_holdout_covgap": mean_holdout_covgap,"mean_tune_covgap":mean_tune_covgap, "mean_tuning_bias":mean_tuning_bias,
                        "mean_holdout_coverage": mean_holdout_coverage, "mean_tune_coverage": mean_tune_coverage,
                        "std_holdout_covgap": std_holdout_covgap, "std_tune_covgap": std_tune_covgap,
                        "std_tuning_bias": std_tuning_bias, "std_holdout_coverage": std_holdout_coverage,
                        "std_tune_coverage": std_tune_coverage
                        }
    save_exp_result(args, mean_result_dict, path=f"./experiment/{args.algorithm}/mean_result")
    print("Mean Result")
    print("mean_tuning_bias: ", mean_tuning_bias)
    print("mean_holdout_covgap: ", mean_holdout_covgap)
    print("mean_tune_covgap: ", mean_tune_covgap)

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
    trainer.train(train_loader, 200)
    save_model(args, trainer.model)
    args.load = "True"
