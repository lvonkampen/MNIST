import os
import torch
import torch.nn as nn
import torchaudio.transforms as T

class Hyperparameters:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024
    train_shuffle, val_shuffle = False, False
    in_feat, hidden_feat, out_feat = 40, [1024, 256, 128], 1
    conv_channels = [32, 64]

    initial_learning_rate = 0.01
    epochs = 5000
    patience = 2

    loss_func = nn.MSELoss()
    optim_func = torch.optim.Adam
    activation = nn.Sigmoid()

    sample_rate = 16000
    transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=in_feat,
        melkwargs={"n_mels": 64, "n_fft": 400, "hop_length": 160}
    )

    max_examples = 5

    max_duration = 300.0    # seconds

    automerge = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(script_dir, "raudio")
    ann_dir = os.path.join(script_dir, "csv_annotations_shardul")
    parent_dir = 'C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/'

    label_map = {
        "relevant":1.0,
        "partially-relevant":0.5,
        "irrelevant":0.0
    }

    iou_threshold = 0.50

# should be a json file so that it is easier to modify
