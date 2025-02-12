import os
import torch
import torch.nn as nn

import torchaudio
import torchaudio.transforms
import torchaudio.transforms as transforms_audio
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Dataset

import matplotlib.pyplot as plt
from random import randint
from AudioMNISTModel import AudioMnistModel


class FlatAudioDataset(Dataset):
    def __init__(self, filepaths, labels, sample_rate=16000, fixed_length=1.0):
        self.filepaths = filepaths
        self.labels = labels
        self.sample_rate = sample_rate
        self.fixed_length = int(sample_rate * fixed_length)  # Fixed length in samples

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(filepath)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if waveform.size(1) > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]
        elif waveform.size(1) < self.fixed_length:
            padding = self.fixed_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform, label

def load_audio_filepaths_and_labels(root):
    filepaths = []
    labels = []
    for filename in os.listdir(root):
        if filename.endswith(".wav"):  # Assuming all files are .wav
            filepath = os.path.join(root, filename)
            label = int(filename.split("_")[0])  # Assuming the label is the first part of the filename
            filepaths.append(filepath)
            labels.append(label)
    return filepaths, labels

def evaluate(model, val_loader, loss_func):
    outputs = [model.validation_step(batch, loss_func) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, model, train_loader, val_loader, optim, loss_func):
    history = []
    for epoch in range(epochs):
        ## Training Phase
        for batch in train_loader:
            optim.zero_grad()
            loss = model.training_step(batch, loss_func)
            loss.backward()
            optim.step()

        ## Validation phase
        result = evaluate(model, val_loader, loss_func)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim = 1)
    return preds[0].item()

def learning_curve(model, train_loader, val_loader, initial_learning_rate, epochs, loss_func):
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    history = []

    for epoch in range(epochs * 10):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.training_step(batch, loss_func)
            loss.backward()
            optimizer.step()

        result = evaluate(model, val_loader, loss_func)
        history.append(result['val_loss'])
        print(f"Epoch {epoch + 1}/{epochs * 10}, Validation Loss: {result['val_loss']:.4f}")

    plt.plot(history, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Learning Curve (Loss vs. Epochs)')
    plt.show()

def training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim):
    learning_rates = [initial_learning_rate, initial_learning_rate / 10, initial_learning_rate / 100, initial_learning_rate / 1000, initial_learning_rate / 10000]
    current_learning_index = 0
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader, optim, loss_func)
        val_result, val_accuracy = model.validate_epoch(val_loader, loss_func)

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


def testing_examples(model, test_dataset):
    for i in range(0, 5):
        img, label = test_dataset[randint(0, len(test_dataset)-1)]
        plt.imshow(img[0], cmap='gray')
        prediction = predict_image(img, model)
        print('Label:', label, '- Predicted:', prediction, '✔' if label==prediction else '✖')

def main():
    root = 'C:/Users/GoatF/Downloads/AI_Practice/AudioMNIST/recordings/'
    filepaths, labels = load_audio_filepaths_and_labels(root)

    # HYPERPARAMETERS
    batch_size = 4096 * 16
    train_shuffle, val_shuffle = True, False
    initial_learning_rate = 0.01
    epochs = 100
    patience = 5
    loss_func = nn.CrossEntropyLoss()
    optim_func = torch.optim.Adam
    activation = nn.ReLU()
    sample_rate = 16000
    n_mels = 64
    n_fft = 1024
    hop_length = 512
    transform = transforms_audio.MelSpectrogram(sample_rate, n_mels, n_fft, hop_length) # equivalent to Tensor

    audio_dataset = FlatAudioDataset(filepaths, labels, sample_rate)
    train_size = int(0.8 * len(audio_dataset))
    val_size = int(0.2 * len(audio_dataset))
    train_data, validation_data = random_split(audio_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size, train_shuffle)
    val_loader = DataLoader(validation_data, batch_size, val_shuffle)

    example_model = AudioMnistModel(activation, 64*32, 32, 10)
    model = AudioMnistModel(activation, 64*32, 32, 10)

    optim = optim_func(model.parameters(), lr=initial_learning_rate)

    # learning_curve(example_model, train_loader, val_loader, initial_learning_rate, 5, loss_func)
    training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim)

    testing_examples(model, audio_dataset)
    result = evaluate(model, val_loader, loss_func)  # Pass loss_func to evaluate()
    print(f"Test Loss: {result['val_loss']:.4f}, Test Accuracy: {result['val_acc']:.2f}%")


if __name__ == "__main__":
    main()