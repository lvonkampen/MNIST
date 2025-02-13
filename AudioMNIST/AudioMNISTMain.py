import wave as wav
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.array_api import float32


class AudioProcessor:
    def __init__(self):
        # Initialize lists to store stats for each file processed
        self.t_shape = []
        self.t_min = []
        self.t_max = []
        self.t_mean = []
        self.t_std = []

    def process_audio(self, filepath):
        with wav.open(filepath, "rb") as wave_file:
            n_frames = wave_file.getnframes()
            frames_buff = wave_file.readframes(n_frames)
            frames_int = np.frombuffer(frames_buff, dtype=np.int16)
            frames_float = frames_int.astype(dtype=float32)

            print(f"Processing: {filepath}")

            shape_ = frames_float.shape
            self.t_shape.append(shape_)
            print("Shape:", shape_)

            print("First 30 samples:", frames_float[:30])

            min_val = frames_float.min()
            print("Min:", min_val)

            max_val = frames_float.max()
            print("Max:", max_val)

            mean_val = frames_float.mean()
            print("Mean:", mean_val)

            std_val = frames_float.std()
            print("Std Dev:", std_val)

            frames_norm = frames_float / std_val

            min_val = frames_norm.min()
            max_val = frames_norm.max()
            mean_val = frames_norm.mean()

            self.t_min.append(min_val)
            self.t_max.append(max_val)
            self.t_mean.append(mean_val)
            self.t_std.append(std_val)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.join(script_dir, "recordings")

    processor = AudioProcessor()

    for filename in os.listdir(recordings_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(recordings_dir, filename)
            processor.process_audio(filepath)

    print("\n--- Summary of Processed Files ---")
    print("Normalized Shapes:", processor.t_shape)
    print("Min Normalized Values:", processor.t_min)
    print("Max Normalized Values:", processor.t_max)
    print("Normalized Means:", processor.t_mean)
    print("Standard Deviation:", processor.t_std)

    print("\n--- Final Summary ---")
    print("Minimum Normalized Value:", min(processor.t_min))
    print("Mean Min Normalized Value:", sum(processor.t_min)/len(processor.t_min))
    print("Maximum Min Normalized Value:", max(processor.t_min))
    print("Minimum Max Normalized Value:", min(processor.t_max))
    print("Mean Max Normalized Value:", sum(processor.t_max)/len(processor.t_max))
    print("Maximum Normalized Value:", max(processor.t_max))
    print("Minimum Normalized Mean Value:", min(processor.t_mean))
    print("Mean Normalized Mean Value:", sum(processor.t_mean)/len(processor.t_mean))
    print("Maximum Normalized Mean Value:", max(processor.t_mean))

    lengths = [shape_tuple[0] for shape_tuple in processor.t_shape]

    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=10, edgecolor='black')
    plt.title("Histogram of Audio Sizes")
    plt.xlabel("Number of Samples")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()