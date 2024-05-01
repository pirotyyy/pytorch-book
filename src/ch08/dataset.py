import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# DATA_ROOT = "/Users/hiroki/dev/university/kawa-kera-lab/workspace/data"
DATA_ROOT = "/home/hiroki/dev/datasets"


def load_data(batch_size):
    mnist_data_path = os.path.join(DATA_ROOT, "MNIST")
    download = not os.path.exists(mnist_data_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    trainset = datasets.MNIST(
        root=DATA_ROOT, train=True, download=download, transform=transform
    )

    testset = datasets.MNIST(
        root=DATA_ROOT, train=False, download=download, transform=transform
    )

    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size, shuffle=True)

    return (trainloader, testloader)
