import os
import csv
import json
import time
import uuid
import subprocess
import pandas as pd
import torch
import torchaudio

from Model import WhisperVIAModel
from Config import Hyperparameters
from Main import testing_examples, split_speakers, CustomWhisperVIADataset, collate_fn, DataLoader

model = WhisperVIAModel(Hyperparameters.activation, Hyperparameters.hidden_feat, Hyperparameters.in_feat,
                        Hyperparameters.out_feat, Hyperparameters.conv_channels)

train_sp, val_sp, test_sp = split_speakers(Hyperparameters.audio_dir)
test_ds = CustomWhisperVIADataset(Hyperparameters.audio_dir, Hyperparameters.ann_dir, Hyperparameters.transform, speakers_include=test_sp)
test_loader = DataLoader(test_ds, Hyperparameters.batch_size, shuffle=False, collate_fn=collate_fn)


def load_model(path, device):
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def map_to_relevance(label: int) -> str:
    if label == 0:          return "irrelevant"
    elif label == 0.5:        return "partially-relevant"
    elif label == 1:                   return "relevant"
    else:                   raise Exception('unknown label')

def extract_audio(wav, start_s, end_s, out_wav):
    cmd = ["ffmpeg","-y","-i",wav,
           "-ss",str(start_s),"-to",str(end_s),
           "-ar","16000","-ac","1", out_wav]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_inference_from_via_csv(via_csv, wav_path, model, transform, device):
    df = pd.read_csv(via_csv, quotechar='"', skiprows=9)

    # Build initial VIA project JSON structure
    base_name = os.path.splitext(os.path.basename(via_csv))[0]
    proj = {
        "project": {
            "pid": "__VID__",
            "pname": base_name,
            "created": int(time.time() * 1000),
            "vid_list": ["1"]
        },
        "config": {
            "file": {"loc_prefix": {"1": ""}},
            "ui": {}
        },
        "attribute": {
            "1": {
                "aname": "WhisperVIA",
                "type": 4,
                "options": {
                    "Irrelevant": "Irrelevant",
                    "Partially-Relevant": "Partially-Relevant",
                    "Relevant": "Relevant"
                },
                "default_option_id": "Irrelevant"
            }
        },
        "file": {
            "1": {
                "fid": "1",
                "fname": os.path.basename(wav_path),
                "type": 4,
                "loc": 1,
                "src": ""
            }
        },
        "metadata": {},
        "view": {
            "1": {
                "fid_list": ["1"]
            }
        }
    }

    tmp_wav = "__tmp.wav"

    for _, row in df.iterrows():
        try:
            # Parse temporal coordinates from string like "[start, end]"
            start, end = json.loads(row["temporal_coordinates"])
        except:
            continue  # skip rows with malformed temporal data

        extract_audio(wav_path, start, end, tmp_wav)

        wf, _ = torchaudio.load(tmp_wav)
        mfcc = transform(wf).to(device)

        with torch.no_grad():
            label_idx = torch.argmax(model(mfcc), dim=1).item()

        label = map_to_relevance(label_idx)
        mid = f"1_{uuid.uuid4().hex[:8]}"

        proj["metadata"][mid] = {
            "vid": "1",
            "flg": 0,
            "z": [start, end],
            "xy": [],
            "av": {
                "1": label
            }
        }

    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)


if __name__=="__main__":

    csv = 'C:\\Users\\GoatF\\Downloads\\AI_Practice\\WhisperVIA\\csv_annotations\\00000_000_001_annotation.csv'
    wav = 'C:\\Users\\GoatF\\Downloads\\AI_Practice\\WhisperVIA\\raudio\\00000_000_001.wav'
    model_path = 'C:\\Users\\GoatF\\Downloads\\AI_Practice\\WhisperVIA\\WhisperVIAModel_NEW.pth'

    model = load_model(model_path, Hyperparameters.device)

    testing_examples(model, test_loader, Hyperparameters.max_examples, Hyperparameters.device)
    # FINISH WORKING ON TESTING EXAMPLES TO PROVIDE TEMPORAL_COORDS + ASSOCIATED PREDS

    run_inference_from_via_csv(csv, wav, model, Hyperparameters.transform, Hyperparameters.device)
