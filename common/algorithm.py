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

        if args.save == "True":
            save_exp_result(args, final_result_dict)

def standard(args):
    for run in args.num_runs:
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