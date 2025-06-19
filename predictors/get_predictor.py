from .cluster_predictor import ClusterPredictor
from .predictor import Predictor
from .localized_predictor import LocalizedPredictor
from .conditional_predictor import ConditionalPredictor
from .condconf_predictor import CondConfPredictor

def get_predictor(args, net):
    if args.predictor == "local":
        predictor = LocalizedPredictor(args, net)
    elif args.predictor == "cluster":
        predictor = ClusterPredictor(args, net)
    elif args.predictor == "condconf":
        predictor = CondConfPredictor(args, net)
    elif args.predictor == "cond":
        predictor = ConditionalPredictor(args, net)
    elif args.predictor == "naive":
        predictor = Predictor(args, net)
    return predictor