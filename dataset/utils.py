import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from .rxrx1 import rxrx1_dataset

def build_train_dataloader(args):
    dataset_name = args.dataset

    if dataset_name == "cifar10":
        train_dataset = CIFAR10(root='./data/dataset', train=True, download=False,
                                transform=transforms.Compose([transforms.ToTensor()]))
        num_classes = 10
        train_dataset = Subset(train_dataset, range(0, 200))
    elif dataset_name == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = CIFAR100(root="/mnt/sharedata/ssd3/common/datasets/cifar-100-python", download=False, train=True, transform=train_transform)
        num_classes = 100
        train_dataset = Subset(train_dataset, range(100, 600))
    elif dataset_name == "imagenet":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load datasets
        train_dataset = torchvision.datasets.ImageFolder(
            root="/mnt/sharedata/ssd3/common/datasets/imagenet/images/train",
            transform=train_transform
        )
        num_classes = 1000
    elif dataset_name == "rxrx1":
        train_dataset = rxrx1_dataset(train=True)
        num_classes = 1039
    else:
        raise NotImplementedError
    args.num_classes = num_classes
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader


def build_cal_test_loader(args):
    dataset_name = args.dataset

    if dataset_name == "cifar10":
        num_classes = 10
        val_dataset = CIFAR10(root='./data/dataset', train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name == "cifar100":
        num_classes = 100

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        val_dataset = CIFAR100(root='/mnt/sharedata/ssd3/common/datasets/cifar-100-python', download=False, train=False,
                                 transform=val_transform)

    elif dataset_name == "imagenet":
        num_classes = 1000

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_dataset = torchvision.datasets.ImageFolder(
            root="/mnt/sharedata/ssd3/common/datasets/imagenet/images/val",
            transform=val_transform
        )
    elif args.dataset == "rxrx1":
        val_dataset = rxrx1_dataset(train=False)
        num_classes = 1039
    else:
        raise NotImplementedError

    if args.algorithm == "standard":
        cal_loader, tune_loader= None, None
        test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.algorithm == "tune":
        cal_size = args.cal_num
        tune_size = args.tune_size if args.tune_num else cal_size
        test_size = len(val_dataset) - cal_size
        cal_dataset, test_dataset = random_split(val_dataset, [cal_size, test_size])
        tune_dataset, test_dataset = random_split(test_dataset, [tune_size, test_size - tune_size])

        cal_dataset = Subset(val_dataset, range(0, 100))
        tune_dataset = Subset(val_dataset, range(100, 200))
        test_dataset = Subset(val_dataset, range(200, 300))
        print(len(tune_dataset))

        cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)
        tune_loader = DataLoader(tune_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=max(args.batch_size, 100), shuffle=False)
    else:
        cal_size = args.cal_num
        test_size = len(val_dataset) - cal_size
        cal_dataset, test_dataset = random_split(val_dataset, [cal_size, test_size])


        cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)
        tune_loader = None
        test_loader = DataLoader(test_dataset, batch_size=max(args.batch_size, 100), shuffle=False)

    args.num_classes = num_classes
    return cal_loader, tune_loader, test_loader


def split_dataloader(original_dataloader, split_ratio=0.5):
        """
        Splits a DataLoader into two Datasets

        Args:
            original_dataloader (DataLoader): The original DataLoader to split.
            split_ratio (float): The ratio of the first subset (default: 0.5).

        Returns:
            subset1: Training dataset
            subset2: Calibration dataset
        """
        dataset = original_dataloader.dataset
        total_size = len(dataset)

        split_size = int(split_ratio * total_size)

        indices = torch.randperm(total_size)
        indices_subset1 = indices[:split_size]
        indices_subset2 = indices[split_size:]

        subset1 = Subset(dataset, indices_subset1)
        subset2 = Subset(dataset, indices_subset2)

        return subset1, subset2
