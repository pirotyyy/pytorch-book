from dataset import load_data
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    # load data
    X, y = load_data()

    # データ前処理
    X, y = X - X.mean(), y - y.mean()

    # to tensor
    X, y = torch.tensor(X), torch.tensor(y)

    # prepare parameter
    W, b = torch.tensor(1.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)

    # prepare pred func
    def pred(X):
        return W * X + b

    # prepare loss_fn
    def mse(y_pred, y):
        loss = ((y_pred - y) ** 2).mean()
        return loss

    n_epochs = 1000

    lr = 0.001

    histories = np.zeros((0, 2))

    for epoch in range(n_epochs):
        y_pred = pred(X)

        loss = mse(y_pred, y)

        loss.backward()

        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad

            W.grad.zero_()
            b.grad.zero_()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.4f}")
            history = np.array([epoch, loss.item()])
            histories = np.vstack((histories, history))

    plt.plot(histories[:, 0], histories[:, 1], "b")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
