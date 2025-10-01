# Libraries
from scipy.io import loadmat, savemat
import os
import shutil
import numpy as np

# Input and output directories
input_directory = r"C:\Users\Usuario\OneDrive - Universidad Industrial de Santander\Database_new"
output_directory = r"C:\Users\Usuario\OneDrive - Universidad Industrial de Santander\DATABASESUB"

# Sampling configuration
fs_original = 500
fs_new = 250
factor = fs_original // fs_new


def downsample_by_average(signal, factor):
    """
    Downsample ECG signal by averaging blocks of size 'factor'.
    Keeps the shape consistent for multi-lead signals.
    """
    new_length = signal.shape[1] // factor
    signal = signal[:, :new_length * factor]  # Trim to multiple of factor
    signal_ds = signal.reshape(signal.shape[0], new_length, factor).mean(axis=2)
    return signal_ds


def update_header_frequency(header_file, new_frequency, num_samples):
    """
    Modify the first line of the .hea file to update sampling frequency
    and number of samples, keeping the rest of the metadata unchanged.
    """
    with open(header_file, 'r') as f:
        lines = f.readlines()

    parts = lines[0].strip().split()
    if len(parts) >= 4:
        parts[2] = str(new_frequency)
        parts[3] = str(num_samples)
        lines[0] = ' '.join(parts) + '\n'

    with open(header_file, 'w') as f:
        f.writelines(lines)


# Main processing loop
for file_name in os.listdir(input_directory):
    if file_name.endswith(".mat"):
        # Load ECG signal from .mat file
        path_mat = os.path.join(input_directory, file_name)
        data = loadmat(path_mat)
        signal = data["val"]

        # Downsample signal
        signal_ds = downsample_by_average(signal, factor)

        # Save new .mat file with downsampled signal only
        new_mat_path = os.path.join(output_directory, file_name)
        savemat(new_mat_path, {"val": signal_ds}, format='4')

        # Copy and update corresponding .hea file
        base_name = file_name.replace('.mat', '')
        hea_src = os.path.join(input_directory, base_name + ".hea")
        hea_dst = os.path.join(output_directory, base_name + ".hea")

        try:
            shutil.copy(hea_src, hea_dst)
            update_header_frequency(hea_dst, fs_new, signal_ds.shape[1])
        except Exception as e:
            print(f"Error processing {base_name}: {e}")