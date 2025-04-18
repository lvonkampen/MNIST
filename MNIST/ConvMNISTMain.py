import time

import torch
import torch.nn as nn
from random import randint

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

from ConvMNISTModel import ConvMnistModel, accuracy


def evaluate(model, val_loader, loss_func, device):
    outputs = [model.validation_step(batch, loss_func) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, model, train_loader, val_loader, optim, loss_func, device):
    history = []
    for epoch in range(epochs):
        ## Training Phase
        for batch in train_loader:
            optim.zero_grad()
            loss = model.training_step(batch, loss_func)
            loss.backward()
            optim.step()

        ## Validation phase
        result = evaluate(model, val_loader, loss_func, device)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def predict_image(img, model, device):
    xb = img.unsqueeze(0).to(device)
    yb = model(xb)
    _, preds = torch.max(yb, dim = 1)
    return preds[0].item()

def learning_curve(model, train_loader, val_loader, initial_learning_rate, epochs, loss_func, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    history = []

    for epoch in range(epochs * 10):
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = model.training_step(batch, loss_func)
            loss.backward()
            optimizer.step()

        result = evaluate(model, val_loader, loss_func, device)
        history.append(result['val_loss'])
        print(f"Epoch {epoch + 1}/{epochs * 10}, Validation Loss: {result['val_loss']:.4f}")

    plt.plot(history, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Learning Curve (Loss vs. Epochs)')
    plt.show()

def training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim, device):
    learning_rates = [initial_learning_rate, initial_learning_rate / 10, initial_learning_rate / 100, initial_learning_rate / 1000, initial_learning_rate / 10000]
    current_learning_index = 0
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader, optim, loss_func,device)
        val_result, val_accuracy = model.validate_epoch(val_loader, loss_func, device)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_result['val_loss']:.6f}")
        print(f"Accuracy: {val_accuracy:.2f}%")

        if val_result['val_loss'] < best_loss:
            best_loss = val_result['val_loss']
            patience_counter = 0
            print(f" - Validation Loss Improved To: {val_result['val_loss']:.6f}")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            current_learning_index += 1  # Move to the next learning rate stage
            if current_learning_index < len(learning_rates):
                new_lr = learning_rates[current_learning_index]
                for param_group in optim.param_groups:
                    param_group['lr'] = new_lr
                print(f"Reducing learning rate to {new_lr:.6e} and resetting patience.")
                patience_counter = 0  # Reset patience
            else:
                print(f"Stopping At Epoch: {epoch + 1} - Best Validation Loss: {best_loss:.6f}")
                break


def testing_examples(model, test_dataset, device):
    for i in range(0, 5):
        img, label = test_dataset[randint(0, len(test_dataset)-1)]
        plt.imshow(img[0], cmap='gray')
        # plt.show()
        prediction = predict_image(img, model, device)
        print('Label:', label, '- Predicted:', prediction, '✔' if label==prediction else '✖')

def timed_training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim, device):
    start_time = time.time()
    training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim, device)
    total_time = time.time() - start_time
    print(f"Total Elapsed Time: ", total_time)

def main():

    # HYPERPARAMETERS
    batch_size = 1024
    train_shuffle, val_shuffle = True, False
    in_feat, hidden_feat, out_feat = 28*28, [256, 128], 10
    conv_channels = [32, 64]
    initial_learning_rate = 0.01
    epochs = 1000
    patience = 5
    loss_func = nn.CrossEntropyLoss()
    optim_func = torch.optim.Adam
    activation = nn.ReLU()

    mnist_dataset = MNIST(root = 'data/', download = True)
    train_dataset = MNIST(root = 'data/', train = True, transform = transforms.ToTensor())
    test_dataset = MNIST(root = 'data/', train = False, transform = transforms.ToTensor())

    train_data, validation_data = random_split(train_dataset, [50000, 10000])

    train_loader = DataLoader(train_data, batch_size, train_shuffle)
    val_loader = DataLoader(validation_data, batch_size, val_shuffle)
    test_loader = DataLoader(test_dataset, batch_size)

    # example_model = ConvMnistModel(activation, hidden_feat, [256, 128])
    model = ConvMnistModel(activation, hidden_feat, out_feat, conv_channels)

    device = torch.device("cuda")
    # example_model.to(device)
    model.to(device)

    optim = optim_func(model.parameters(), lr=initial_learning_rate)

    # learning_curve(example_model, train_loader, val_loader, initial_learning_rate, 5, loss_func, device)
    timed_training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim, device)

    testing_examples(model, test_dataset, device)
    result = evaluate(model, test_loader, loss_func, device)  # Pass loss_func to evaluate()
    print(f"Test Loss: {result['val_loss']:.4f}, Test Accuracy: {result['val_acc']:.2f}%")

if __name__ == "__main__":
    main()

