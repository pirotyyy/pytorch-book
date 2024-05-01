import torch
import torch.nn as nn
from dataset import load_data
from model import CNN
from trainer import Trainer


def main():
    # set manual seed
    torch.manual_seed(42)

    # load_data
    train_loader, test_loader = load_data(batch_size=128)

    # gen model
    model = CNN(10)
    model.cuda()

    # learning rate
    lr = 0.01

    # optimizer 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # n_epochs
    n_epochs = 100

    trainer = Trainer(model, optimizer, criterion)

    trainer.train(train_loader, test_loader, n_epochs)



if __name__ == '__main__':
    main()