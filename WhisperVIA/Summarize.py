import subprocess
import torch
import os
import pandas as pd
import ast
import torchaudio
import matplotlib.pyplot as plt

from moviepy import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips

from Config import Hyperparameters
from Model import WhisperVIAModel

def video_to_whisper(vid, wav, out_dir):
    # Extract audio from video using ffmpeg
    print(f"Extracting audio from {vid}")
    subprocess.run([
        "ffmpeg", "-i", vid, "-y", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav
    ], check=True)
    # Transcribe using Whisper CLI
    print(f"Transcribing audio from {wav}")
    subprocess.run([
        "whisper", wav, "--model", "medium.en", "--output_dir", out_dir])

def run_inference(tsv, wav, transform, device, model, sample_rate):
    # Parse Whisper TSV
    df = pd.read_csv(tsv, sep='\t')
    df = df[['start', 'end', 'text']].astype({'start': float, 'end': float})

    info = torchaudio.info(wav)
    audio_dur_s = info.num_frames / info.sample_rate
    print(f"Audio duration: {audio_dur_s:.2f}s")

    max_time = max(df['start'].max(), df['end'].max())
    df['start_ms'] = df['start'].astype(int)
    df['end_ms'] = df['end'].astype(int)

    results = []

    for _, row in df.iterrows():
        start_ms, end_ms = row.start_ms, row.end_ms
        offset = int((start_ms / 1000) * sample_rate)
        length = int(((end_ms - start_ms) / 1000) * sample_rate)
        waveform, _ = torchaudio.load(wav, frame_offset=offset, num_frames=length)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mfcc = transform(waveform).to(device)  # [1, n_mels, T]
        with torch.no_grad():
            score = model(mfcc).squeeze().item()
        results.append((start_ms, end_ms, score))
    return results

def concat_segments(segments, max_duration, vid, summary, automerge):
    segments.sort(key=lambda x: x[2], reverse=True)

    # Pick top‑scoring segments up to max_duration
    chosen, total = [], 0.0
    for start_ms, end_ms, score in segments:
        length = end_ms - start_ms
        if total + length > max_duration:
            continue
        chosen.append((start_ms, end_ms, score))
        total += length
    print(f"Picking {len(chosen)} clips : {total/1000:.1f}s total")

    # Resort them in temporal order
    chosen = sorted(chosen, key=lambda x: x[0])

    if automerge:
        chosen = [list(seg) for seg in chosen]
        merged = []
        prev = chosen[0]
        for curr in chosen[1:]:
            # if total + length > max_duration:
            #     continue
            gap = curr[0] - prev[1]
            if (gap) <= Hyperparameters.merge_gap * 1000:
                prev[1] = curr[1]
                prev[2] = max(prev[2], curr[2])
            else:
                length = prev[1] - prev[0]
                total  += length
                merged.append(prev)
                prev    = curr
        length = prev[1] - prev[0]
        total  += length
        merged.append(prev)
        chosen = merged
        print(f"Merged to {len(chosen)} clips : {total / 1000:.1f}s total")


    # Load the full video once
    full_video = VideoFileClip(vid)

    clips = []
    for start_ms, end_ms, score in chosen:
        # Trim by time using `subclipped`
        clip = full_video.subclipped(start_ms/1000, end_ms/1000)
        # Create a top‑right TextClip with the confidence score
        txt = TextClip(text=f"Confidence: {score:.2f}",
                       font_size=28,
                       color='white',
                       bg_color='black',
                       text_align="center",
                       horizontal_align="right",
                       vertical_align="top",
                       duration=clip.duration)
        # Composite the text over the video clip
        comp = CompositeVideoClip([clip, txt])
        clips.append(comp)

    # Concatenate all annotated clips into one summary
    final = concatenate_videoclips(clips)
    final.write_videofile(summary)

    print(f"Summary written to {summary}")

parent = Hyperparameters.parent_dir

def cleanup(vid_id):
    os.remove(f"{parent}WhisperVIASummarization/{vid_id}.json")
    os.remove(f"{parent}WhisperVIASummarization/{vid_id}.srt")
    os.remove(f"{parent}WhisperVIASummarization/{vid_id}.tsv")
    os.remove(f"{parent}WhisperVIASummarization/{vid_id}.txt")
    os.remove(f"{parent}WhisperVIASummarization/{vid_id}.vtt")
    os.remove(f"{parent}WhisperVIASummarization/{vid_id}.wav")

def summarize(vid_id):
    vid_path        = f"C:/Git_Repositories/AccessMath/data/original_videos/lectures/{vid_id}.mp4"
    wav_path        = f"{parent}WhisperVIASummarization/{vid_id}.wav"
    whisper_out_dir = f"{parent}WhisperVIASummarization/"
    tsv_path        = f"{parent}WhisperVIASummarization/{vid_id}.tsv"
    model_path      = f"{parent}WhisperVIAModel.pth"
    summary_path    = f"{parent}WhisperVIASummarization/{vid_id}_summary_automerge.mp4" if Hyperparameters.automerge else f"{parent}WhisperVIASummarization/{vid_id}_summary.mp4"

    model = WhisperVIAModel(Hyperparameters.activation, Hyperparameters.hidden_feat, Hyperparameters.in_feat,
                            Hyperparameters.out_feat, Hyperparameters.conv_channels)

    state_dict = torch.load(model_path, map_location=Hyperparameters.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(Hyperparameters.device).eval()

    video_to_whisper(vid_path, wav_path, whisper_out_dir)
    #segments = run_inference(tsv_path, wav_path, Hyperparameters.transform, Hyperparameters.device, model, Hyperparameters.sample_rate)
    #concat_segments(segments, Hyperparameters.max_duration * 1000, vid_path, summary_path, Hyperparameters.automerge)
    #cleanup(vid_id)


def parse_via_annotations(path, label_map):
    df = pd.read_csv(path, comment='#', header=None,
                     names=["id", "file_list", "flags", "temporal_coordinates", "spatial_coordinates", "metadata"])

    # Clean columns
    df["temporal_coordinates"] = df["temporal_coordinates"].apply(ast.literal_eval)
    df["metadata"] = df["metadata"].apply(ast.literal_eval)

    # Extract label and score
    def extract_label(meta):
        label = meta.get("1", None)  # "1" is the attribute ID from the header
        return label.lower() if label else None

    df["label"] = df["metadata"].apply(extract_label)
    df = df[df["label"].isin(label_map)]  # filter out Whisper or unknowns

    # Convert to (start_ms, end_ms, score)
    segments = []
    for _, row in df.iterrows():
        coords = row["temporal_coordinates"]
        if len(coords) == 2:
            start_ms = int(coords[0] * 1000)
            end_ms = int(coords[1] * 1000)
            score = label_map[row["label"]]
            segments.append((start_ms, end_ms, score))
    return segments

def sum_ground_truth(vid_id):
    vid_path        = f"C:/Git_Repositories/AccessMath/data/original_videos/lectures/{vid_id}.mp4"
    ann_path        = f"{parent}csv_annotations_shardul/{vid_id}_annotation.csv" # some annotations are called _annotation while others are _annotations !
    summary_path    = f"{parent}WhisperVIASummarization/{vid_id}_ground_truth.mp4"

    segments = parse_via_annotations(ann_path, Hyperparameters.label_map)

    concat_segments(segments, Hyperparameters.max_duration * 1000, vid_path, summary_path, False)

    cleanup(vid_id)

def make_histogram(vid_id):
    # segments must be sorted by start time
    ann_path = f"{parent}csv_annotations_shardul/{vid_id}_annotation_shardul.csv"

    segments = parse_via_annotations(ann_path, Hyperparameters.label_map)

    segs = sorted(segments, key=lambda x: x[0])
    gaps = []
    for (s1,e1,_), (s2,_,_) in zip(segs, segs[1:]):
        gaps.append((s2 - e1)/1000.0)  # in seconds

    plt.figure()
    plt.hist(gaps, bins=30)      # no color specs!
    plt.xlabel("Gap duration (s)")
    plt.ylabel("Count")
    plt.title("Histogram of Inter‐segment Gaps")
    plt.show()

def main():
    vid_id = "00026_000_001"
    summarize(vid_id)
    # sum_ground_truth(vid_id)
    #make_histogram(vid_id)

if __name__ == "__main__":
    main()


# seperate parts of the script to add modularity
# research how to use command line in debugging mode : Run - Edit configurations - > apply configuration <

# see how to transition audios by overlaying
# change relevance annotations - think of reducing the video by 10x