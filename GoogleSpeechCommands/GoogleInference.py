import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import os
from GoogleModel import GoogleModel

# HYPERPARAMETERS
activation = nn.ReLU()
in_feat, hidden_feat, out_feat = 40, [256, 128, 64], 10
conv_channels = [32, 64]

def load_model(model_path, device):
    model = GoogleModel(activation, hidden_feat, in_feat, out_feat, conv_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_inference_audio(file_path, transform):
    waveform, sample_rate = torchaudio.load(file_path)
    mfcc = transform(waveform)
    mfcc = mfcc.squeeze(0)
    return mfcc

def run_inference(model, inference_dir, transform, device):
    print("\n--- Running Inference on Inference-Scripts Folder ---")

    for filename in os.listdir(inference_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(inference_dir, filename)
            inference_audio = load_inference_audio(file_path, transform)
            prediction = predict_image(inference_audio, model, device)
            print(f"File: {filename} → Predicted Label: {prediction}")

def predict_image(img, model, device):
    xb = img.unsqueeze(0).to(device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("GoogleSpeechCommandsModel.pth", device)
    inference_dir = "C:/Users/GoatF/Downloads/AI_Practice/GoogleSpeechCommandsModel/Inference-Scripts" # NOT EXISTING YET

    transform = T.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={"n_mels": 64, "n_fft": 400, "hop_length": 160})

    run_inference(model, inference_dir, transform, device)

if __name__ == "__main__":
    main()

#