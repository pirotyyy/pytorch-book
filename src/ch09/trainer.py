from tqdm import tqdm
import torch
import numpy as np

class Trainer():
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = np.zeros((0, 5))

    def get_history(self):
        return self.history

    def reset_history(self):
        self.history = np.zeros((0, 5))

    def train(self, train_loader, test_loader, n_epochs=100):
        base_epochs = len(self.history)

        for epoch in range(base_epochs, n_epochs + base_epochs):
            n_train_acc, n_test_acc = 0, 0
            train_loss, test_loss = 0, 0
            n_train, n_test = 0, 0
            

            # Training
            self.model.train()

            for x_train, y_train in tqdm(train_loader):
                train_batch_size = len(x_train)
                n_train += train_batch_size

                x_train, y_train = x_train.cuda(), y_train.cuda()

                outputs = self.model(x_train)

                loss = self.criterion(outputs, y_train)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                pred = torch.max(outputs, 1)[1]

                train_loss += loss.item() * train_batch_size
                n_train_acc += (pred == y_train).sum().item()

            # Testing
            self.model.eval()

            for x_test, y_test in test_loader:
                test_batch_size = len(y_test)
                n_test += test_batch_size

                x_test = x_test.cuda()
                y_test = y_test.cuda()

                with torch.inference_mode():
                    outputs_test = self.model(x_test)

                    loss_test = self.criterion(outputs_test, y_test)

                    predicted_test = torch.max(outputs_test, 1)[1]

                    test_loss += loss_test.item() * test_batch_size
                    n_test_acc += (predicted_test == y_test).sum().item()

            train_acc = n_train_acc / n_train
            test_acc = n_test_acc / n_test

            avg_train_loss = train_loss / n_train
            avg_test_loss = test_loss / n_test

            print(f"Epoch: [{epoch + 1}/{n_epochs+base_epochs}], Train Loss: {avg_train_loss:.5f}, Train Acc: {train_acc:.5f}, Test Loss: {avg_test_loss:.5f}, Test Acc: {test_acc:.5f}")

            item = np.array([epoch + 1, avg_train_loss, train_acc, avg_test_loss, test_acc])
            history = np.vstack((self.history, item))

        return history
    