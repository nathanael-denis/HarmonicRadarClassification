import os
import json
import numpy as np
import random
import scipy.signal as signal
import cv2
from PIL import Image

# Parameters
IMAGE_SIZE = (224, 224)
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
SEED = 42

AUGMENT_TRAIN = True
AUGMENT_OOD = True
SNR_TRAIN_RANGE = (0, 5)    # dB
SNR_OOD_RANGE = (-5, 0)      # dB

WINDOW_SIZE = 256
OVERLAP = 128
NFFT = 256
FS = 10e6
SAMPLES_PER_SEGMENT = 100000

random.seed(SEED)
np.random.seed(SEED)


# ---------------- AWGN ----------------
def add_awgn(iq_samples, snr_db):
    signal_power = np.mean(np.abs(iq_samples) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(iq_samples)) + 1j * np.random.randn(len(iq_samples))
    )
    return iq_samples + noise


# ---------------- Spectrogram ----------------
def iq_to_spectrogram(iq_samples, window_size=256, overlap=128, nfft=256,
                      window_type='hann', image_size=(224, 224), fs=100000):
    window = signal.get_window(window_type, window_size)
    f, t, Zxx = signal.stft(
        iq_samples, fs=fs, window=window,
        nperseg=window_size, noverlap=overlap, nfft=nfft
    )

    # Power in dB
    power_spectrogram = np.abs(Zxx) ** 2
    power_spectrogram_db = 10 * np.log10(power_spectrogram + 1e-12)

    # Contrast enhancement
    min_val = np.percentile(power_spectrogram_db, 1)
    max_val = np.percentile(power_spectrogram_db, 99)
    clipped = np.clip(power_spectrogram_db, min_val, max_val)

    # Normalize 0–255
    norm = (clipped - min_val) / (max_val - min_val + 1e-12)
    spectrogram = (norm * 255).astype(np.uint8)

    # Resize to model input
    spectrogram = cv2.resize(spectrogram, image_size, interpolation=cv2.INTER_LINEAR)
    return spectrogram


# ---------------- Image Writing ----------------
def write_spectrogram(iq_segment, split_name, out_folder,
                      window_size=WINDOW_SIZE, overlap=OVERLAP, nfft=NFFT,
                      image_size=IMAGE_SIZE, fs=FS):
    # Add noise depending on split
    if split_name == 'train' and AUGMENT_TRAIN:
        snr_db = np.random.uniform(*SNR_TRAIN_RANGE)
        iq_segment = add_awgn(iq_segment, snr_db)
    elif split_name == 'ood' and AUGMENT_OOD:
        snr_db = np.random.uniform(*SNR_OOD_RANGE)
        iq_segment = add_awgn(iq_segment, snr_db)

    # Build spectrogram
    spectrogram = iq_to_spectrogram(
        iq_segment, window_size, overlap, nfft,
        image_size=image_size, fs=fs
    )

    os.makedirs(out_folder, exist_ok=True)
    fname = f"spec_{np.random.randint(1e8):08d}.png"
    Image.fromarray(spectrogram).save(os.path.join(out_folder, fname))


# ---------------- Main Dataset Processing ----------------
def process_iq_dataset(iq_dir, output_dir):
    classes = sorted([d for d in os.listdir(iq_dir) if os.path.isdir(os.path.join(iq_dir, d))])

    # Save mapping for training
    class_to_idx = {c: i for i, c in enumerate(classes)}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    for class_name in classes:
        class_path = os.path.join(iq_dir, class_name)
        iq_files = sorted([
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ])

        for iq_path in iq_files:
            print(f"Processing {iq_path}")
            iq = np.fromfile(iq_path, dtype=np.complex64)

            # Split into segments
            total_segments = len(iq) // SAMPLES_PER_SEGMENT
            segments = [
                (i * SAMPLES_PER_SEGMENT, (i + 1) * SAMPLES_PER_SEGMENT)
                for i in range(total_segments)
            ]

            # Shuffle once for reproducibility
            random.Random(SEED).shuffle(segments)

            n = len(segments)
            n_train = int(n * TRAIN_SPLIT)
            n_val = int(n * VAL_SPLIT)

            idx_train = segments[:n_train]
            idx_val = segments[n_train:n_train + n_val]
            idx_test = segments[n_train + n_val:]

            # Write splits
            for (s, e) in idx_train:
                out = os.path.join(output_dir, "train", class_name)
                write_spectrogram(iq[s:e], "train", out)

            for (s, e) in idx_val:
                out = os.path.join(output_dir, "val", class_name)
                write_spectrogram(iq[s:e], "val", out)

            for (s, e) in idx_test:
                out = os.path.join(output_dir, "test", class_name)
                write_spectrogram(iq[s:e], "test", out)

            # OOD augmentation only from test set
            if AUGMENT_OOD:
                for (s, e) in idx_test:
                    out = os.path.join(output_dir, "ood", class_name)
                    write_spectrogram(iq[s:e], "ood", out)


if __name__ == "__main__":
    IQ_DIR = r"C:\GNURadio\CruzerBlade"
    OUTPUT_DIR = r"output"
    process_iq_dataset(IQ_DIR, OUTPUT_DIR)
