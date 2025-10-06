import os
import numpy as np
import scipy.signal as signal
import cv2
import random  # Add this import for random number generation
from PIL import Image

# Convert IQ samples to grayscale spectrogram image
def iq_to_spectrogram(iq_samples, window_size=256, overlap=128, nfft=256,
                      window_type='hann', image_size=(224, 224), fs=10e6):
    window = signal.get_window(window_type, window_size)
    f, t, Zxx = signal.stft(iq_samples, fs=fs, window=window,
                            nperseg=window_size, noverlap=overlap, nfft=nfft)
    # Use power (in dB) for better contrast
    power_spectrogram = np.abs(Zxx)**2
    power_spectrogram_db = 10 * np.log10(power_spectrogram + 1e-12)
    # Clip to remove extreme values
    min_val = np.percentile(power_spectrogram_db, 1)
    max_val = np.percentile(power_spectrogram_db, 99)
    clipped = np.clip(power_spectrogram_db, min_val, max_val)
    # Normalize to 0–255
    norm = (clipped - min_val) / (max_val - min_val + 1e-12)
    spectrogram = (norm * 255).astype(np.uint8)
    # Resize to model input size
    spectrogram = cv2.resize(spectrogram, image_size, interpolation=cv2.INTER_LINEAR)
    return spectrogram  # Single-channel grayscale image

# Generate and save spectrograms (no augmentation)
def save_spectrograms(iq_samples, samples_per_segment=10000, window_size=256,
                      overlap=128, nfft=256, window_type='hann', fs=10e6,
                      image_size=(224, 224), output_folder='images'):
    num_samples = len(iq_samples)
    num_segments = num_samples // samples_per_segment
    os.makedirs(output_folder, exist_ok=True)
    for i in range(num_segments):
        start_idx = i * samples_per_segment
        end_idx = start_idx + samples_per_segment
        segment = iq_samples[start_idx:end_idx]
        if len(segment) < window_size:
            continue
        spectrogram = iq_to_spectrogram(
            segment, window_size, overlap, nfft, window_type, image_size, fs
        )
        # Generate a random number for the filename
        random_number = random.randint(10000000, 99999999)
        filename = os.path.join(output_folder, f'image_{i:04d}_{random_number}.png')
        Image.fromarray(spectrogram).save(filename)

# Process all IQ files recursively
def process_all_iq_files(base_directory='RawIQ', window_size=256, overlap=128, nfft=256,
                         window_type='hann', samples_per_segment=10000,
                         image_size=(224, 224), fs=10e6,
                         output_base='images'):
    for subdir, _, files in os.walk(base_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            print(f'Processing file: {file_path}')
            iq_samples = np.fromfile(file_path, dtype=np.complex64)
            relative_subdir = os.path.relpath(subdir, base_directory)
            output_folder = os.path.join(output_base, relative_subdir)
            save_spectrograms(
                iq_samples=iq_samples,
                samples_per_segment=samples_per_segment,
                window_size=window_size,
                overlap=overlap,
                nfft=nfft,
                window_type=window_type,
                image_size=image_size,
                fs=fs,
                output_folder=output_folder
            )

# Entry point
if __name__ == "__main__":
    current_dir = os.getcwd()
    process_all_iq_files(
        base_directory=os.path.join(current_dir, 'C:\\GNURadio\\OrderedSamples\\Sandisk ultra deluxe 256GB'),
        window_size=256,
        overlap=128,
        nfft=256,
        samples_per_segment=100000,
        image_size=(224, 224),
        fs=10e6,
        output_base=os.path.join(current_dir, 'images')
    )
