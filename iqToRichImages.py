import os
import numpy as np
from scipy.fft import fftshift, fft
from PIL import Image
import random

def compute_spectrum(iq_samples, fft_size=1024):
    spectrum = np.abs(fftshift(fft(iq_samples, fft_size)))
    spectrum_db = 20 * np.log10(spectrum + 1e-6)  # Avoid log of zero
    spectrum_db -= np.min(spectrum_db)
    spectrum_db /= np.max(spectrum_db)
    spectrum_db *= 255
    return spectrum_db

def save_stacked_spectra_as_images(iq_samples, fft_size=1024, image_size=(224, 224), num_stacks=1024, output_folder='images'):
    # Ensure fft_size and num_stacks are integers
    fft_size = int(fft_size)
    num_stacks = int(num_stacks)
    
    num_samples = len(iq_samples)
    num_chunks = num_samples // fft_size
    stacked_spectrum = np.zeros((num_stacks, fft_size))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i in range(num_chunks // num_stacks):
        for j in range(num_stacks):
            start_idx = (i * num_stacks + j) * fft_size
            chunk = iq_samples[start_idx:start_idx + fft_size]
            if len(chunk) < fft_size:
                break
            spectrum_db = compute_spectrum(chunk, fft_size)
            stacked_spectrum[j, :] = spectrum_db
        
        # Reshape and normalize the stacked spectrum to fit into an image
        stacked_image = stacked_spectrum.flatten()
        stacked_image -= np.min(stacked_image)
        stacked_image /= np.max(stacked_image)
        stacked_image *= 255
        stacked_image = stacked_image.reshape((num_stacks, fft_size))
        
        # Convert to image and resize
        image = Image.fromarray(stacked_image.astype(np.uint8))
        image = image.resize(image_size, Image.BILINEAR)
        random_number = random.randint(10000000, 99999999)
        image_filename = os.path.join(output_folder, f'image_{i:04d}_{random_number}.png')
        image.save(image_filename)

def process_all_iq_files(base_directory='rawIQ', fft_size=1024, image_size=(224, 224), output_base='images'):
    # Iterate over all subdirectories in the base directory
    for subdir, _, files in os.walk(base_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            print(f'Processing file: {file_path}')
            
            # Load the IQ samples
            iq_samples = np.fromfile(file_path, dtype=np.complex64)
            
            # Determine output directory by preserving the directory structure
            relative_subdir = os.path.relpath(subdir, base_directory)
            output_folder = os.path.join(output_base, relative_subdir)
            
            # Generate and save images
            save_stacked_spectra_as_images(iq_samples=iq_samples, 
                                           fft_size=fft_size, 
                                           image_size=image_size, 
                                           num_stacks=128, 
                                           output_folder=output_folder)

# Call the function to process all files using current directory
current_dir = os.getcwd()
base_directory = os.path.join(current_dir, 'rawIQ')
output_base = os.path.join(current_dir, 'images')

process_all_iq_files(base_directory=base_directory, 
                     fft_size=4096, 
                     image_size=(224, 224), 
                     output_base=output_base)