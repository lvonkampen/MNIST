import os
import time
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

import wave as wav
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from sklearn.metrics import precision_score, recall_score, f1_score

from Dataset import CustomWhisperVIADataset
from Model import WhisperVIAModel

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_t = max(x.shape[1] for x in xs)
    xs_padded = [
        nn.functional.pad(x, (0, max_t - x.shape[1]), mode='constant', value=0)
        for x in xs
    ]
    x_batch = torch.stack(xs_padded, dim=0)
    y_batch = torch.tensor(ys, dtype=torch.float32)
    return x_batch, y_batch

def evaluate(model, val_loader, loss_func, device):
    total_loss = 0.0
    total_samples = 0
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            yb = yb.float().view(-1, 1)
            out = model(xb)

            loss = loss_func(out, yb)
            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

            preds = out.view(-1).cpu().numpy()
            labels = yb.view(-1).cpu().numpy()

            # Quantize predictions and cast to integers (0, 1, 2)
            quantized_preds = [int(quantize(p) * 2) for p in preds]  # 0.0 → 0, 0.5 → 1, 1.0 → 2
            quantized_labels = [int(l * 2) for l in labels]  # same for labels

            all_preds.extend(quantized_preds)
            all_labels.extend(quantized_labels)

    avg_loss = total_loss / total_samples

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Evaluation Metrics → Loss: {avg_loss:.6f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return avg_loss, precision, recall, f1

def timed_training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim, device):
    start_time = time.time()
    train_loss, val_loss = training(model, train_loader, val_loader, loss_func, initial_learning_rate, epochs, patience, optim, device)
    total_time = time.time() - start_time
    print(f"Total Elapsed Time: {(total_time/60)/60:.2f} hours")
    return train_loss, val_loss

def plot_spectrogram(spectrogram, label):
    plt.figure(figsize=(8, 6))
    plt.imshow(spectrogram, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label="Intensity (dB)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.title(f"Spectrogram - Label: {label}")
    plt.show()

def quantize(prediction):
    if prediction < 0.34:   return 0.0
    if prediction < 0.67:   return 0.5
    return 1.0

def testing_examples(model, test_dataset, max_examples, device):
    model.eval()
    tested = 0
    predictions_sequenced = []
    with torch.no_grad():
        for xb, yb in test_dataset:
            xb, yb = xb.to(device), yb.to(device).float()
            tested += 1
            if tested >= max_examples:
                return
        for img, label in zip(xb, yb):
            raw_prediction = predict_image(img, model, device)
            predictions_sequenced.append(raw_prediction)
            prediction = quantize(raw_prediction)
            print(f'Label:', label.item(), '- Predicted:', prediction, '\t', '✔' if label == prediction else '✖', ' - Raw Prediction:', raw_prediction)
    return predictions_sequenced


def predict_image(img, model, device):
    xb = img.unsqueeze(0).to(device)
    yb = model(xb)
    return yb.item()

def split_speakers(audio_dir, train_perc, val_perc, seed, save_path, force_new):
    if os.path.exists(save_path) and not force_new:
        print(f"Loading speaker split from: {save_path}")
        with open(save_path, 'r') as f:
            split = json.load(f)
        return split['train'], split['val'], split['test']

    speakers = set()
    for fn in os.listdir(audio_dir):
        if fn.endswith(".wav"):
            speakers.add(fn.split("_", 1)[0])

    speakers = list(speakers)
    random.seed(seed)
    random.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * train_perc)
    n_val = int(n * val_perc)

    train = speakers[:n_train]
    val = speakers[n_train:n_train + n_val]
    test = speakers[n_train + n_val:]

    with open(save_path, 'w') as f:
        json.dump({'train': train, 'val': val, 'test': test}, f, indent=4)

    print(f"Saved speaker split to: {save_path}")
    return train, val, test


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

    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader, optim, loss_func, device)
        val_loss = model.validate_epoch(val_loader, loss_func, device)
        val_losses.append(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}\nTrain Loss: {train_loss:.6f}\tValidation Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            print(f" - Validation Loss Improved To: {val_loss:.6f}")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            current_learning_index += 1
            if current_learning_index < len(learning_rates):
                new_lr = learning_rates[current_learning_index]
                for param_group in optim.param_groups:
                    param_group['lr'] = new_lr
                print(f"Reducing learning rate to {new_lr:.6e} and resetting patience.")
                patience_counter = 0
            else:
                print(f"Stopping At Epoch: {epoch + 1} - Best Validation Loss: {best_loss:.6f}")
                break

    return train_losses, val_losses


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


def main():
    with open("config.json", "r") as f:
        hyper = json.load(f)

    device = torch.device(hyper["device"] if torch.cuda.is_available() else "cpu")

    loss_func = getattr(nn, hyper["training"]["loss_func"])()
    optim_func = getattr(torch.optim, hyper["training"]["optim_func"])
    activation = getattr(nn, hyper["model"]["activation"])()

    if hyper["transform"]["type"] == "MFCC":
        transform = T.MFCC(
            sample_rate=hyper["data"]["sample_rate"],
            n_mfcc=hyper["data"]["transform"]["n_mfcc"],
            melkwargs=hyper["data"]["transform"]["melkwargs"]
        )
    else:
        raise ValueError(f"Unknown transform type: {hyper["data"]["transform"]["type"]}")

    train_sp, val_sp, test_sp = split_speakers(
        audio_dir=  hyper["paths"]["audio_dir"],
        train_perc= hyper["split_speaker"]["train_perc"],
        val_perc=   hyper["split_speaker"]["valid_perc"],
        seed=       hyper["split_speaker"]["rand_seed"],
        save_path=  hyper["split_speaker"]["path"],
        force_new=  hyper["split_speaker"]["force_new"]
    )

    train_ds = CustomWhisperVIADataset( hyper["paths"]["audio_dir"], hyper["paths"]["ann_dir"], transform, speakers_include=train_sp)
    val_ds = CustomWhisperVIADataset(   hyper["paths"]["audio_dir"], hyper["paths"]["ann_dir"], transform, speakers_include=val_sp)
    test_ds = CustomWhisperVIADataset(  hyper["paths"]["audio_dir"], hyper["paths"]["ann_dir"], transform, speakers_include=test_sp)

    train_loader = DataLoader(  train_ds, hyper["training"]["batch_size"], shuffle=hyper["training"]["train_shuffle"], collate_fn=collate_fn)
    val_loader = DataLoader(    val_ds,   hyper["training"]["batch_size"], shuffle=hyper["training"]["val_shuffle"], collate_fn=collate_fn)
    test_loader = DataLoader(   test_ds,  hyper["training"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = WhisperVIAModel(
        activation,
        hyper["model"]["hidden_feat"],
        hyper["model"]["in_feat"],
        hyper["model"]["out_feat"],
        hyper["model"]["conv_channels"],
        hyper["model"]["adaptive_pool_size"]
    )
    model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Trainable Parameters in Network:", pytorch_total_params)

    optim = optim_func(model.parameters(), lr=hyper["initial_learning_rate"])

    train_losses, val_losses = timed_training(
        model,
        train_loader,
        val_loader,
        loss_func,
        hyper["training"]["initial_learning_rate"],
        hyper["training"]["epochs"],
        hyper["training"]["patience"],
        optim,
        device
    )

    plot_loss_progression(train_losses, val_losses)

    testing_examples(model, test_loader, hyper["data"]["max_examples"], device)

    test_loss, test_precision, test_recall, test_f1 = evaluate(model, test_loader, loss_func, device)
    print(
        f"\nFinal Test Metrics:\nLoss: {test_loss:.6f}\nPrecision: {test_precision:.4f}\nRecall: {test_recall:.4f}\nF1 Score: {test_f1:.4f}")

    torch.save(model.state_dict(), "WhisperVIAModel_1x0x2.pth")
    print("Model saved as 'WhisperVIAModel_1x0x2.pth'")

if __name__ == "__main__":
    main()

