import os
import re
import json

import pandas as pd
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


with open("config.json", "r") as f:
    cfg = json.load(f)

class CustomWhisperVIADataset(Dataset):
    label_map = cfg["label_map"]
    sample_rate = cfg["sample_rate"]

    TIME_RE = re.compile(r"[0-9]+\.?[0-9]*")

    def __init__(self,
                 audio_dir: str,
                 ann_dir: str,
                 transform=None,
                 speakers_include: list[str] | None = None):
        self.audio_dir = audio_dir
        self.ann_dir   = ann_dir
        self.transform = transform
        self.speakers_include = set(speakers_include) if speakers_include else None

        self.segments = []

        for fn in os.listdir(self.ann_dir):
            if not fn.lower().endswith(".csv"):
                continue
            path = os.path.join(self.ann_dir, fn)

            df = pd.read_csv(
                path,
                comment="#",
                header=None,
                names=[
                    "metadata_id",
                    "file_list",
                    "flags",
                    "temporal_coordinates",
                    "spatial_coordinates",
                    "metadata"
                ],
                dtype=str,
                skip_blank_lines=True
            )

            for idx, row in df.iterrows():
                t_raw = row["temporal_coordinates"].strip()
                times = None
                try:
                    parsed = json.loads(t_raw)
                    if isinstance(parsed, list) and len(parsed) >= 2:
                        times = [float(parsed[0]), float(parsed[1])]
                except Exception:
                    pass
                if times is None:
                    found = self.TIME_RE.findall(t_raw)
                    if len(found) >= 2:
                        times = [float(found[0]), float(found[1])]
                if not times or len(times) < 2:
                    print(f"Warning: Skipping annotation in file '{fn}' (row {idx}) — malformed or missing temporal_coordinates: {t_raw}")
                    continue
                start_s, end_s = times

                fl_raw = row["file_list"].strip()
                wav_base = None
                try:
                    files = json.loads(fl_raw)
                    wav_base = os.path.splitext(files[0])[0]
                except Exception:
                    m = re.search(r'"([^"]+\.mp4)"', fl_raw)
                    if m:
                        wav_base = os.path.splitext(m.group(1))[0]
                if not wav_base:
                    continue

                wav_p = os.path.join(self.audio_dir, wav_base + ".wav")
                if not os.path.exists(wav_p):
                    print("Could not find audio file: " + wav_p + "\nSkipping annotation!! Notify Dr. Davila for entire dataset")
                    continue

                label_str = "irrelevant"
                meta_raw = row["metadata"].strip()
                try:
                    meta = json.loads(meta_raw)
                    label_str = meta.get("1", label_str).lower()
                except Exception:
                    raise Exception("Typo found within file: " + label_str)

                try:
                    label = float(label_str)
                except ValueError:
                    label = self.label_map.get(label_str, -1)
                    if label == -1:
                        raise Exception("Unknown label found: " + label_str)

                self.segments.append((wav_p, start_s, end_s, label))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        wav_p, start_s, end_s, label = self.segments[idx]

        start_frame = int(start_s * self.sample_rate)
        num_frames  = int((end_s - start_s) * self.sample_rate)

        wf, _ = torchaudio.load(wav_p, frame_offset=start_frame, num_frames=num_frames)

        if wf.shape[0] > 1: # Convert to mono from stereo
            wf = wf.mean(dim=0, keepdim=True)

        mfcc = self.transform(wf).squeeze(0)  # [40, T]

        return mfcc, label


def main():
    if cfg["transform"]["type"] == "MFCC":
        transform = T.MFCC(
            sample_rate=cfg["sample_rate"],
            n_mfcc=cfg["transform"]["n_mfcc"],
            melkwargs=cfg["transform"]["melkwargs"]
        )
    else:
        raise ValueError(f"Unknown transform type: {cfg['transform']['type']}")

    dataset = CustomWhisperVIADataset(cfg["audio_dir"], cfg["ann_dir"], transform)
    durations = [end - start for _, start, end, _ in dataset.segments]
    print(f"Min: {min(durations):.2f}s | Max: {max(durations):.2f}s | Avg: {sum(durations) / len(durations):.2f}s")

if __name__ == "__main__":
    main()