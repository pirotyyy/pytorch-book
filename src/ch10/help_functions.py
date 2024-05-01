import matplotlib.pyplot as plt
import numpy as np
import torch

def evaluate_history(history):
    #損失と精度の確認
    print(f'initial_state: loss: {history[0,3]:.5f} acc: {history[0,4]:.5f}') 
    print(f'final_state: loss: {history[-1,3]:.5f} acc: {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='train')
    plt.plot(history[:,0], history[:,3], 'k', label='test')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve (Loss)')
    plt.legend()
    plt.savefig('loss.png')

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='train')
    plt.plot(history[:,0], history[:,4], 'k', label='test')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title('Learning Curve (Accuracy)')
    plt.legend()
    plt.savefig('accuracy.png')

def torch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True