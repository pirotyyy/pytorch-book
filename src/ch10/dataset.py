import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# DATA_ROOT = "/Users/hiroki/dev/university/kawa-kera-lab/workspace/data"
DATA_ROOT = "/home/hiroki/dev/datasets"


def load_data(batch_size):
    cifar10_data_path = os.path.join(DATA_ROOT, "CIFFAR10")
    download = not os.path.exists(cifar10_data_path)

    transform_train = transforms.Compose(
        [
            transforms.Resize(112),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ]
    )

    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    trainset = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=download, transform=transform_train
    )

    testset = datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=download, transform=transform
    )

    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size, shuffle=True)

    return (trainloader, testloader)
