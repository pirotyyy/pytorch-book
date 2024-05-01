import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from dataset import load_data
from trainer import Trainer
from help_functions import evaluate_history, torch_seed


def main():
    # set manual seed
    torch_seed()

    # device
    print('cuda' if torch.cuda.is_available() else 'cpu')

    # load_data
    train_loader, test_loader = load_data(batch_size=50)

    # gen model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    fc_in_features = model.fc.in_features
    model.fc = nn.Linear(fc_in_features, 10)
    model.cuda()


    # optimizer 
    optimizer = torch.optim.Adam(model.parameters())

    # criterion
    criterion = nn.CrossEntropyLoss()

    # n_epochs
    n_epochs = 50

    trainer = Trainer(model, optimizer, criterion)

    trainer.train(train_loader, test_loader, n_epochs)

    evaluate_history(trainer.get_history())



if __name__ == '__main__':
    main()