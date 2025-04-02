import os
import time
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from numba.core.typing.dictdecl import infer
from torch.utils.data import Dataset, DataLoader, Subset

import wave as wav
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from random import randint, shuffle

from AudioMNISTModel import AudioMnistModel

class CustomAudioMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filepaths = []
        self.labels = []
        self.speakers = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".wav"):
                        filepath = os.path.join(folder_path, filename)
                        label = int(filename.split("_")[0])
                        self.filepaths.append(filepath)
                        self.labels.append(label)
                        self.speakers.append(folder)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        file_path = self.filepaths[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        if self.transform:
            mfcc = self.transform(waveform)  # Shape: [1, 40, T]
            mfcc = mfcc.squeeze(0)  # Now shape: [40, T]
        else:
            mfcc = waveform
        return mfcc, label


def convert_m4a_to_wav(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".m4a"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(output_dir, output_filename)

            command = ["C:\\FFmpeg\\ffmpeg.exe", "-i", input_path, output_path]
            print(f"Running command: {command}")
            subprocess.run(command)

            print(f"Converted: {filename} -> {output_filename}")

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs_transposed = [inp.transpose(0, 1) for inp in inputs]  # each becomes [T, 40]
    padded = torch.nn.utils.rnn.pad_sequence(inputs_transposed, batch_first=True, padding_value=0)
    padded = padded.transpose(1, 2)
    return padded, torch.tensor(labels)

def evaluate(model, val_loader, loss_func):
    outputs = [model.validation_step(batch, loss_func) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def timed_training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim, device):
    start_time = time.time()
    train_loss, val_loss, train_accu, val_accu = training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim, device)
    total_time = time.time() - start_time
    print(f"Total Elapsed Time: ", total_time)
    return train_loss, val_loss, train_accu, val_accu

def plot_spectrogram(spectrogram, label):
    plt.figure(figsize=(8, 6))
    plt.imshow(spectrogram, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label="Intensity (dB)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.title(f"Spectrogram - Label: {label}")
    plt.show()

def testing_examples(model, test_dataset, device):
    for i in range(5):
        img, label = test_dataset[randint(0, len(test_dataset) - 1)]
        # plot_spectrogram(img.cpu().numpy(), label)
        prediction = predict_image(img, model, device)
        print('Label:', label, '- Predicted:', prediction, '✔' if label == prediction else '✖')

def predict_image(img, model, device):
    xb = img.unsqueeze(0).to(device)
    yb = model(xb)
    _, preds = torch.max(yb, dim = 1)
    return preds[0].item()

def split_speakers(root_dir, train_perc=0.7, val_perc=0.2):
    speaker_folders = [d for d in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, d))]
    shuffle(speaker_folders)

    total = len(speaker_folders)
    num_train = int(total * train_perc)
    num_val = int(total * val_perc)

    train_speakers = speaker_folders[:num_train]
    val_speakers   = speaker_folders[num_train:num_train + num_val]
    test_speakers  = speaker_folders[num_train + num_val:]

    return train_speakers, val_speakers, test_speakers

class SpeakerDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(path)

        if self.transform:
            mfcc = self.transform(waveform)
            mfcc = mfcc.squeeze(0)
        else:
            mfcc = waveform

        return mfcc, label

def collect_files(root_dir, speakers):
    file_paths = []
    labels = []

    for speaker_folder in speakers:
        folder_path = os.path.join(root_dir, speaker_folder)
        for fname in os.listdir(folder_path):
            if fname.endswith(".wav"):
                path = os.path.join(folder_path, fname)
                label_str = fname.split("_")[0]
                label = int(label_str)
                file_paths.append(path)
                labels.append(label)

    return file_paths, labels

def load_inference_audio(file_path, transform):
    waveform, sample_rate = torchaudio.load(file_path)
    mfcc = transform(waveform)
    mfcc = mfcc.squeeze(0)
    print("File working!")

    return mfcc

def run_inference_on_all(model, inference_dir, transform, device):
    print("\n--- Running Inference on Inference-Scripts Folder ---")

    for filename in os.listdir(inference_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(inference_dir, filename)
            inference_audio = load_inference_audio(file_path, transform)
            prediction = predict_image(inference_audio, model, device)
            print(f"File: {filename} → Predicted Label: {prediction}")

def training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim, device):
    learning_rates = [initial_learning_rate, initial_learning_rate / 10, initial_learning_rate / 100,
                      initial_learning_rate / 1000, initial_learning_rate / 10000]
    current_learning_index = 0
    best_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        train_loss, train_accuracy = model.train_epoch(train_loader, optim, loss_func, device)
        val_loss, val_accuracy = model.validate_epoch(val_loader, loss_func, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss['val_loss'])
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Assuming val_accuracy is in fraction format and needs scaling
        print(f"Epoch {epoch + 1}/{epochs}\nTrain Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%")
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
                for param_group in optim.param_groups:
                    param_group['lr'] = new_lr
                print(f"Reducing learning rate to {new_lr:.6e} and resetting patience.")
                patience_counter = 0  # Reset patience
            else:
                print(f"Stopping At Epoch: {epoch + 1} - Best Validation Loss: {best_loss:.6f}")
                break

    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_loss_progression(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Progression")
    plt.legend()
    plt.show()

def display(processor):
    print("\n--- Summary of Processed Files ---")
    print("Normalized Shapes:", processor.t_shape)
    print("Min Normalized Values:", processor.t_min)
    print("Max Normalized Values:", processor.t_max)
    print("Normalized Means:", processor.t_mean)
    print("Standard Deviation:", processor.t_std)

    print("\n--- Final Summary ---")
    print("Minimum Shape:", min(processor.t_shape))
    print("Maximum Shape:", max(processor.t_shape))
    print("Minimum Normalized Value:", min(processor.t_min))
    print("Mean Min Normalized Value:", sum(processor.t_min) / len(processor.t_min))
    print("Maximum Min Normalized Value:", max(processor.t_min))
    print("Minimum Max Normalized Value:", min(processor.t_max))
    print("Mean Max Normalized Value:", sum(processor.t_max) / len(processor.t_max))
    print("Maximum Normalized Value:", max(processor.t_max))
    print("Minimum Normalized Mean Value:", min(processor.t_mean))
    print("Mean Normalized Mean Value:", sum(processor.t_mean) / len(processor.t_mean))
    print("Maximum Normalized Mean Value:", max(processor.t_mean))

    print("Sample Count >8000:", processor.count_above_8k)
    print("Total Files Processed:", processor.total_files)

    # for file in processor.files_above_8k:
    #     print("Filename >8000:", file)

    lengths = [shape_tuple[0] for shape_tuple in processor.t_shape]

    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=10, edgecolor='black')
    plt.title("Histogram of Audio Sizes")
    plt.xlabel("Number of Samples")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


class AudioProcessor:
    def __init__(self):
        self.t_shape = []
        self.t_min = []
        self.t_max = []
        self.t_mean = []
        self.t_std = []
        self.count_above_8k = 0
        self.files_above_8k = []
        self.total_files = 0

    def process_audio(self, filepath):
        self.total_files += 1
        with wav.open(filepath, "rb") as wave_file:
            sample_rate = wave_file.getframerate()
            n_frames = wave_file.getnframes()
            frames_buff = wave_file.readframes(n_frames)
            frames_int = np.frombuffer(frames_buff, dtype=np.int16)
            frames_float = frames_int.astype(dtype=float)

            print(f"Processing: {filepath}")


            shape_ = frames_float.shape
            min_val = frames_float.min()
            max_val = frames_float.max()
            mean_val = frames_float.mean()
            std_val = frames_float.std()

            if shape_[0] > 8000:
                self.count_above_8k += 1
                self.files_above_8k.append(filepath)

            print(f"Sample rate: {sample_rate} Hz")
            # print("Shape:", shape_)
            # print("First 30 samples:", frames_float[:30])
            # print("Min:", min_val)
            # print("Max:", max_val)
            # print("Mean:", mean_val)
            # print("Std Dev:", std_val)

            frames_norm = frames_float / std_val

            min_val = frames_norm.min()
            max_val = frames_norm.max()
            mean_val = frames_norm.mean()

            self.t_shape.append(shape_)
            self.t_min.append(min_val)
            self.t_max.append(max_val)
            self.t_mean.append(mean_val)
            self.t_std.append(std_val)


def main():

    # HYPERPARAMETERS
    batch_size = 1024
    train_shuffle, val_shuffle = True, False
    in_feat, hidden_feat, out_feat = 40, [256, 128, 64], 10
    conv_channels = [32, 64]
    initial_learning_rate = 0.01
    epochs = 1000
    patience = 5
    loss_func = nn.CrossEntropyLoss()
    optim_func = torch.optim.Adam
    activation = nn.ReLU()
    train_perc = 0.7
    valid_perc = 0.2
    test_perc = 0.1
    transform = T.MFCC(sample_rate=16000,             # Hz
                       n_mfcc=in_feat,                # Number of features
                       melkwargs={"n_mels": 64,       # Bins in spectrogram
                                  "n_fft": 400,       # Affects frequency resolution?
                                  "hop_length": 160}) # Stride between FFT windows --> affects time resolution.

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    # convert_m4a_to_wav(inference_dir, inference_dir) # this will generate .wav duplicates

    dataset = CustomAudioMNISTDataset(root_dir=data_dir, transform=transform)

    train_speakers, val_speakers, test_speakers = split_speakers(data_dir)

    train_files, train_labels = collect_files(data_dir, train_speakers)
    val_files, val_labels = collect_files(data_dir, val_speakers)
    test_files, test_labels = collect_files(data_dir, test_speakers)

    train_data = SpeakerDataset(train_files, train_labels, transform=transform)
    val_data = SpeakerDataset(val_files, val_labels, transform=transform)
    test_data = SpeakerDataset(test_files, test_labels, transform=transform)

    # 5) Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # processor = AudioProcessor()

    # for folder in os.listdir(data_dir):
    #     folder_path = os.path.join(data_dir, folder)
    #     if os.path.isdir(folder_path):
    #         for filename in os.listdir(folder_path):
    #             if filename.endswith(".wav"):
    #                 filepath = os.path.join(folder_path, filename)
    #                 processor.process_audio(filepath)

    # display(processor)

    model = AudioMnistModel(activation, hidden_feat, in_feat, out_feat, conv_channels)
    device = torch.device("cuda")
    model.to(device)

    optim = optim_func(model.parameters(), lr=initial_learning_rate)

    train_losses, val_losses, train_accu, val_accu = timed_training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim, device)

    # plot_loss_progression(train_losses, val_losses)

    testing_examples(model, test_data, device)
    result = evaluate(model, test_loader, loss_func)  # Pass loss_func to evaluate()
    print(f"Test Loss: {result['val_loss']:.4f}, Test Accuracy: {result['val_acc']:.2f}%")

    torch.save(model.state_dict(), "AudioMNISTModel.pth")
    print("Model saved as 'AudioMNISTModel.pth'")

if __name__ == "__main__":
    main()

# NEW DATASETS:

#       UrbanSounds8K          - Contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark...

#       FluentSpeechCommands   - Commands spoken in natural language for virtual assistants (e.g., "Turn the lights on in the kitchen"). 30000 utterances
#                                Labels: Actions(activate, increase, decrease), object(lights, music), and location(kitchen, bedroom).

#       Google Speech Commands - Small dataset of simple voice commands like “yes,” “no,” “left,” “right,” “stop,” “go”. 105000 utterances from 35 different categories
#                              - Labels: Predefined classes like "up," "down," "stop," "go."