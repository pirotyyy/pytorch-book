from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


def main():
    iris = load_iris()

    x_data, y_label = iris.data, iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        x_data[:, [0, 2]], y_label, test_size=0.3, random_state=42
    )

    n_input = x_train.shape[1]
    n_output = len(list(set(y_train)))

    model = IrisModel(n_input, n_output)

    criterion = nn.CrossEntropyLoss()

    lr = 0.01

    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).long()
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).long()

    n_epochs = 10000

    histories = np.zeros((0, 5))

    for epoch in range(n_epochs):
        # Training
        model.train()

        outputs = model(x_train)

        loss = criterion(outputs, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        pred = torch.max(outputs, 1)[1]
        acc = (pred == y_train).sum() / len(y_train)

        # Testing
        model.eval()
        with torch.inference_mode():
            test_outputs = model(x_test)
            test_loss = criterion(test_outputs, y_test)

            test_pred = torch.max(test_outputs, 1)[1]
            test_acc = (test_pred == y_test).sum() / len(y_test)

        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}"
            )

    plot_decision_boundary(model, x_train, y_train)


class IrisModel(nn.Module):
    def __init__(self, n_input, n_output, hidden_size=8):
        super().__init__()

        self.l1 = nn.Linear(n_input, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.l3(self.relu(self.l2(self.relu(self.l1(x)))))


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    input_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    Z = model(input_tensor)
    Z = torch.argmax(Z, dim=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()


if __name__ == "__main__":
    main()
