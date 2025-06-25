import subprocess
import torch
import os
import pandas as pd
import torchaudio

from WhisperVIAConfig import Hyperparameters
from WhisperVIAModel import WhisperVIAModel

def video_to_whisper(vid, wav, out_dir):
    # Extract audio from video using ffmpeg
    print(f"Extracting audio from {vid}")
    subprocess.run([
        "ffmpeg", "-i", vid, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav
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

def concat_segments(segments, max_duration, vid, summary):
    chosen, total = [], 0.0
    for start_ms, end_ms, score in segments:
        length = end_ms - start_ms
        if total + length > max_duration:
            break
        chosen.append((start_ms, end_ms))
        total += length
    total = total / 1000
    print(f"Picking {len(chosen)} clips : {total:.1f}s total")

    # create clips
    clips = []
    for i, (start, end) in enumerate(chosen):
        fn = f"clip_{i:02d}.mp4"
        clips.append(fn)
        subprocess.run([
            "ffmpeg",
            "-ss", f"{start / 1000:.2f}", "-to", f"{end / 1000:.2f}",
            "-i", vid, "-preset", "veryfast", "-crf", "23", # veryfast encodes quickly - more testing
            "-c:v", "libx264", "-c:a", "aac", fn            # necessary to determine optimal speed
        ], check=True)

    # concat clips
    list_txt = "to_concat.txt"
    with open(list_txt, "w", encoding="utf-8") as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")

    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_txt,
        "-preset", "slow", "-crf", "23",
        "-c:v", "libx264", "-c:a", "aac",
        summary
    ], check=True)

    for txt in clips + [list_txt]:
        os.remove(txt)

    print(f"Summary written to {summary}")


def cleanup(vid_id):
    os.remove(f"C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIASummarization/{vid_id}.json")
    os.remove(f"C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIASummarization/{vid_id}.srt")
    os.remove(f"C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIASummarization/{vid_id}.tsv")
    os.remove(f"C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIASummarization/{vid_id}.txt")
    os.remove(f"C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIASummarization/{vid_id}.vtt")
    os.remove(f"C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIASummarization/{vid_id}.wav")

def main():
    vid_id = "00000_000_002"

    vid_path        = f"C:/Git_Repositories/AccessMath/data/original_videos/lectures/{vid_id}.mp4"
    wav_path        = f"C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIASummarization/{vid_id}.wav"
    whisper_out_dir = "C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIASummarization/"
    tsv_path        = f"C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIASummarization/{vid_id}.tsv"
    model_path      = "C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIAModel_NEW.pth"
    summary_path    = f"C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/WhisperVIASummarization/{vid_id}_summary.mp4"

    model = WhisperVIAModel(Hyperparameters.activation, Hyperparameters.hidden_feat, Hyperparameters.in_feat,
                            Hyperparameters.out_feat, Hyperparameters.conv_channels)

    state_dict = torch.load(model_path, map_location=Hyperparameters.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(Hyperparameters.device).eval()

    video_to_whisper(vid_path, wav_path, whisper_out_dir)

    segments = run_inference(tsv_path, wav_path, Hyperparameters.transform, Hyperparameters.device, model, Hyperparameters.sample_rate)
    segments = sorted(segments, key=lambda x: x[2], reverse=True)

    concat_segments(segments, Hyperparameters.max_duration * 1000, vid_path, summary_path)

    cleanup(vid_id)

if __name__ == "__main__":
    main()

# summarize using inference -- take video, hyperparameters, and segments and output a new
# lecture video with 5 minutes worth of the most relevant segments (in descending order)

# make a parent path - make a Hyperparameters path
# make it possible to give transcript from outside source (csv or json files)
# instead of breaking / you should just skip segments
# observe distance of phrases on the video for better summary
# ####  resort the chosen clips before concatenating - sort by start_ms
# ####  make it possible to run summary on ground truth the same way
# you may not need to re-encode audio - see if you can find ways to auto-encode
# look into adding overlay of video (probability of relevance) in Python without much extra processing - OpenCV: last resort
# MUST implement inference (recall / F1 / )