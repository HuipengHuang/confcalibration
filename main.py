import argparse
from common import algorithm


parser = argparse.ArgumentParser()

parser.add_argument("--num_runs", type=int, default=1, help="Number of runs")
parser.add_argument("--model", type=str, default="resnet18", help='Choose neural network architecture.')
parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "imagenet", "rxrx1"],
                    help="Choose dataset for training.")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument("--pretrained", default="True", type=str, choices=["True", "False"])
parser.add_argument("--save", default="False", choices=["True", "False"], type=str)
parser.add_argument("--algorithm",'-alg', default="tune", choices=["standard", "cp", "tune"],
                    help="Uncertainty aware training use uatr. Otherwise use standard")
parser.add_argument("--load", default="False", type=str, choices=["True", "False"])
parser.add_argument("--predictor", default="naive", type=str, choices=["naive", "local", "cluster", "cond", "condconf"])
parser.add_argument("--save_model", default=None, type=str, choices=["True", "False"])
parser.add_argument("--train_one_model_first", default="False", type=str, choices=["True", "False"])
parser.add_argument("--split", default="True", type=str, choices=["True", "False"])

#Tuning bias hyperparameter
parser.add_argument("--method", default="ts", type=str, choices=["ts", "vs", "ps","conftr", "linear_probing"])
parser.add_argument("--cal_num", type=int, default=1000, help="6k, 7k, 8k, 9k, 10k are used in the tuning bias paper")
parser.add_argument("--tune_num", type=int, default=None)
parser.add_argument("--freeze_num", type=int, default=None)

#  Training configuration
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Choose optimizer.")
parser.add_argument("--learning_rate", "-lr", type=float, default=1e-1, help="Initial learning rate for optimizer")
parser.add_argument("--epochs", '-e', type=int, default=None, help="Number of epochs to train. None means dont train")
parser.add_argument("--batch_size",'-bsz', type=int, default=32)
parser.add_argument("--momentum", type=float, default=0, help='Momentum')
parser.add_argument("--weight_decay", type=float, default=0, help='Weight decay')
parser.add_argument("--loss", type=str,default='ce', choices=['ce', 'conftr', 'ua', "cadapter", "hinge"],
                    help='Loss function you want to use. standard loss is Cross Entropy Loss.')

#  Hyperpatameters for Conformal Prediction
parser.add_argument("--alpha", type=float, default=0.1, help="Error Rate")
parser.add_argument("--score", type=str, default="thr", choices=["thr", "aps", "raps", "saps", "weight_score"])

#  Hyperparameters for ConfTr
parser.add_argument("--size_loss_weight", type=float, default=None, help='Weight for size loss in ConfTr')
parser.add_argument("--tau", type=float, default=None,
                    help='Hyperparameter for ConfTr. Soft predicted Size larger than tau will be penalized in the size loss.')
parser.add_argument("--temperature",'-T', type=float, default=None,
                    help='Temperature scaling for ConfTr or C-adapter loss')

#  Hyperparameter for aps, raps and saps
parser.add_argument("--random",type=str,default=None,choices=["True","False"])
parser.add_argument("--raps_size_regularization",type=float, default=10, help='K_reg for raps loss')
parser.add_argument("--raps_weight",type=float, default=1 ,help="lambda for size regularization in raps.")
parser.add_argument("--saps_weight",type=float, default=1 ,help="lambda for size regularization in saps.")

#  Hyperparameter for uncertainty aware loss
parser.add_argument("--mu", type=float, default=None,
                    help="Weight of train_loss_score in the uncertainty_aware_loss function")
parser.add_argument("--mu_size", type=float, default=None,
                    help="Weight of train_loss_size in the uncertainty_aware_loss function")

# Hyperparameter for clustered CP
parser.add_argument("--k", type=int, default=None, help="Number of cluster center in kmeans algorithm")
parser.add_argument("--null_qhat", default="standard", type=str, help="If standard, use standard calibration for classses belong to null.")


args = parser.parse_args()

if args.algorithm == "standard":
    algorithm.standard(args)
elif args.algorithm =="cp":
    algorithm.cp(args)
elif args.algorithm == "tune":
    algorithm.tune(args)

