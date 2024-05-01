import torch
import torch.nn as nn
from dataset import load_data
from model import MNISTModel
import numpy as np
from tqdm import tqdm


def main():
    # set manual seed
    torch.manual_seed(42)

    # set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    # load data
    train_loader, test_loader = load_data(batch_size=128)

    # gen model
    model = MNISTModel()
    model.to(device)

    # learning rate
    lr = 0.01

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # n_epochs
    n_epochs = 100

    # histories
    histories = np.zeros((0, 5))

    for epoch in range(n_epochs):
        # Training
        # 1エポックあたりの正解数(精度計算用)
        n_train_acc, n_val_acc = 0, 0
        # 1エポックあたりの累積損失(平均化前)
        train_loss, val_loss = 0, 0
        # 1エポックあたりのデータ累積件数
        n_train, n_test = 0, 0
        for inputs, labels in tqdm(train_loader):
            train_batch_size = len(labels)
            n_train += train_batch_size

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            pred = torch.max(outputs, 1)[1]

            train_loss += loss.item() * train_batch_size
            n_train_acc += (pred == labels).sum().item()

        # Testing
        for inputs_test, labels_test in test_loader:
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels_test)
            # 1エポックあたりのデータ累積件数
            n_test += test_batch_size

            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            model.eval()
            with torch.inference_mode():
                # 予測計算
                outputs_test = model(inputs_test)

                # 損失計算
                loss_test = criterion(outputs_test, labels_test)

                # 予測ラベル導出
                predicted_test = torch.max(outputs_test, 1)[1]

                #  平均前の損失と正解数の計算
                # lossは平均計算が行われているので平均前の損失に戻して加算
                val_loss += loss_test.item() * test_batch_size
                n_val_acc += (predicted_test == labels_test).sum().item()

        # 精度計算
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        # 損失計算
        ave_train_loss = train_loss / n_train
        ave_val_loss = val_loss / n_test
        # 結果表示
        print(
            f"Epoch [{epoch+1}/{n_epochs}], loss: {ave_train_loss:.5f} acc: {train_acc:.5f} val_loss: {ave_val_loss:.5f}, val_acc: {val_acc:.5f}"
        )
        # 記録
        history = np.array(
            [epoch + 1, ave_train_loss, train_acc, ave_val_loss, val_acc]
        )
        histories = np.vstack((histories, history))


if __name__ == "__main__":
    main()
