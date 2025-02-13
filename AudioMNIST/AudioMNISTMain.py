import wave as wav
import numpy as np
import matplotlib as plt
import os

from numba import float32


def process_audio(filepath):
    with wav.open(filepath, "rb") as wave:
        n_frames = wave.getnframes()
        frames_buff = wave.readframes(n_frames)
        frames_int = np.frombuffer(frames_buff, dtype=np.int16)
        frames_float = frames_int.astype(dtype=float32)

        print(f"Processing: {filepath}")
        print("Shape:", frames_float.shape)
        print("First 30 samples:", frames_float[:30])
        print("Min:", frames_float.min())
        print("Max:", frames_float.max())
        print("Mean:", frames_float.mean())
        print("Stand-Dev:", frames_float.std())

        print("----------------------------------")

        frames_norm = frames_float / frames_float.std()

        print("Normalized Min:", frames_norm.min())
        print("Normalized Max:", frames_norm.max())
        print("Normalized Mean:", frames_norm.mean())


def main():
    directory = os.path.join(os.path.dirname(__file__), "AudioMNIST", "recordings")

    print("Recordings directory:", directory)
    print("Files:", os.listdir(directory))

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            process_audio(filepath)



    if __name__ == "__main__":
        main()