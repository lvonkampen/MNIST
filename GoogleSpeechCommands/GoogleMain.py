import os
import time
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from random import randint, shuffle

from torchaudio.datasets import SPEECHCOMMANDS

from GoogleModel import GoogleModel


def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs_transposed = [inp.transpose(0, 1) for inp in inputs]
    padded = torch.nn.utils.rnn.pad_sequence(inputs_transposed, batch_first=True, padding_value=0)
    padded = padded.transpose(1, 2)
    return padded, torch.tensor(labels)


def plot_loss_progression(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Progression")
    plt.legend()
    plt.show()


def predict_image(img, model, device):
    xb = img.unsqueeze(0).to(device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


def evaluate(model, val_loader, loss_func):
    outputs = [model.validation_step(batch, loss_func) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim_func, device):
    learning_rates = [initial_learning_rate, initial_learning_rate / 10,
                      initial_learning_rate / 100, initial_learning_rate / 1000,
                      initial_learning_rate / 10000]
    current_learning_index = 0
    best_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    optimizer = optim_func(model.parameters(), lr=initial_learning_rate)

    for epoch in range(epochs):
        train_loss, train_accuracy = model.train_epoch(train_loader, optimizer, loss_func, device)
        val_loss, val_accuracy = model.validate_epoch(val_loader, loss_func, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss['val_loss'])
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss['val_loss']:.6f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

        if val_loss['val_loss'] < best_loss:
            best_loss = val_loss['val_loss']
            patience_counter = 0
            print(f" - Validation Loss Improved To: {val_loss['val_loss']:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            current_learning_index += 1  # Move to the next learning rate stage
            if current_learning_index < len(learning_rates):
                new_lr = learning_rates[current_learning_index]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Reducing learning rate to {new_lr:.6e} and resetting patience.")
                patience_counter = 0  # Reset patience
            else:
                print(f"Stopping At Epoch: {epoch + 1} - Best Validation Loss: {best_loss:.6f}")
                break

    return train_losses, val_losses, train_accuracies, val_accuracies


def timed_training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim_func,
                   device):
    start_time = time.time()
    train_loss, val_loss, train_accu, val_accu = training(model, train_loader, val_loader, loss_func,
                                                          initial_learning_rate, epochs, patience, optim_func, device)
    total_time = time.time() - start_time
    print("Total Elapsed Time: ", total_time)
    return train_loss, val_loss, train_accu, val_accu


def testing_examples(model, test_dataset, device, label_map):
    print("\n--- Running Example Predictions on Test Data ---")
    for i in range(5):
        img, label = test_dataset[randint(0, len(test_dataset) - 1)]
        prediction = predict_image(img, model, device)
        real_label = label_map[label]
        predicted_label = label_map[prediction]
        result = "✔" if label == prediction else "✖"
        print(f"Label: {real_label} - Predicted: {predicted_label} {result}")




class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, download: bool = True, data_path: str = "./SpeechCommands"):
        super().__init__(root=data_path, download=download)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as file_obj:
                return [os.path.join(self._path, line.strip()) for line in file_obj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


class SpeechCommandsDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.dataset = SubsetSC(subset=subset)
        self.transform = transform
        if subset == "training":
            self.classes = sorted(list(set(datapoint[2] for datapoint in self.dataset)))
        else:
            train_set = SubsetSC(subset="training")
            self.classes = sorted(list(set(datapoint[2] for datapoint in train_set)))
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, *_ = self.dataset[idx]
        if self.transform:
            mfcc = self.transform(waveform)
            mfcc = mfcc.squeeze(0)
        else:
            mfcc = waveform
        label_idx = self.label_to_index[label]
        return mfcc, label_idx


def main():
    batch_size = 1024
    in_feat = 40
    hidden_feat = [256, 128, 64]
    conv_channels = [32, 64]
    initial_learning_rate = 0.01
    epochs = 100
    patience = 2
    loss_func = nn.CrossEntropyLoss()
    optim_func = torch.optim.Adam
    activation = nn.ReLU()

    transform = T.MFCC(sample_rate=16000, n_mfcc=in_feat,
                       melkwargs={"n_mels": 64, "n_fft": 400, "hop_length": 160})

    train_dataset = SpeechCommandsDataset(subset="training", transform=transform)
    val_dataset = SpeechCommandsDataset(subset="validation", transform=transform)
    test_dataset = SpeechCommandsDataset(subset="testing", transform=transform)

    label_map = {idx: label for label, idx in train_dataset.label_to_index.items()}
    out_feat = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GoogleModel(activation, hidden_feat, in_feat, out_feat, conv_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, val_losses, train_accu, val_accu = timed_training(
        model, train_loader, val_loader, loss_func, initial_learning_rate,
        epochs, patience, optim_func, device
    )

    plot_loss_progression(train_losses, val_losses)

    testing_examples(model, test_dataset, device, label_map)

    result = evaluate(model, test_loader, loss_func)
    print(f"Test Loss: {result['val_loss']:.4f}, Test Accuracy: {result['val_acc'] * 100:.2f}%")

    torch.save(model.state_dict(), "GoogleSpeechCommandsModel.pth")
    print("Model saved as 'GoogleSpeechCommandsModel.pth'")


if __name__ == "__main__":
    main()

# use scikit to label recall / precision / F1