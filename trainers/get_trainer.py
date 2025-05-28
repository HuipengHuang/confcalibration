from .trainer import Trainer
from .uncertainty_aware_trainer import UncertaintyAwareTrainer
def get_trainer(args):
    if args.algorithm == "uatr":
        trainer = UncertaintyAwareTrainer(args)
    else:
        trainer = Trainer(args)
    return trainer
