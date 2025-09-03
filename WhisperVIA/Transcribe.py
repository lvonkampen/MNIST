import os

from Summarize import video_to_whisper
import json


def main():
    # traverses every file in the lectures folder and converts it into a whisper transcript

    with open("config.json", "r") as f:
        hyper = json.load(f)

    parent = hyper["paths"]["parent_dir"]
    input_dir = 'C:/Git_Repositories/AccessMath/data/original_videos/lectures'
    output_dir = 'C:/Users/GoatF/Downloads/AI_Practice/WhisperVIA/whisper_trans'

    for filename in os.listdir(input_dir):
        vid_id, ext = os.path.splitext(filename)
        video_path = os.path.join(input_dir, filename)
        wav_path = os.path.join(output_dir, f"{vid_id}.wav")

        video_to_whisper(video_path, wav_path, output_dir)

        os.remove(f"{parent}whisper_trans/{vid_id}.json")
        os.remove(f"{parent}whisper_trans/{vid_id}.srt")
        os.remove(f"{parent}whisper_trans/{vid_id}.txt")
        os.remove(f"{parent}whisper_trans/{vid_id}.vtt")
        os.remove(f"{parent}whisper_trans/{vid_id}.wav")


if __name__ == "__main__":
    main()