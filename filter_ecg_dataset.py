# Libraries
import os
import shutil
import numpy as np
from scipy.io import loadmat


# Input and output directories
# Input contains the raw PhysioNet-format database (.hea + .mat files)
# Output will store only the patients that satisfy all filtering conditions
input_dir = r"C:\Users\Usuario\OneDrive - Universidad Industrial de Santander\Database"
output_dir = r"C:\Users\Usuario\OneDrive - Universidad Industrial de Santander\Database_new"


# Load all patients in the input directory
def load_all_patients(input_dir):
    header_files = []
    # Collect all header files (.hea) in the directory tree
    for root, _, files in os.walk(input_dir):
        for f in files:
            g = os.path.join(root, f)
            if f.lower().endswith(".hea") and os.path.isfile(g):
                header_files.append(g)

    # Extract diagnostic classes and prepare counters
    classes = get_classes(header_files)
    num_classes = len(classes)
    num_files = len(header_files)
    recordings, headers = [], []

    # Load paired signal and header for each patient
    for hf in header_files:
        signal, header = load_patient(hf)
        recordings.append(signal)
        headers.append(header)

    return recordings, headers, classes, num_classes, num_files


# Load a single patient: read header metadata and paired ECG signal
def load_patient(header_file):
    with open(header_file, "r") as f:
        header = f.readlines()

    # Each .hea file has a paired .mat file with the actual ECG signal
    mat_file = header_file.replace(".hea", ".mat")
    x = loadmat(mat_file)
    signal = np.asarray(x["val"], dtype=np.float64)

    return signal, header


# Extract all unique diagnostic codes from headers
def get_classes(header_files):
    classes = set()
    for filename in header_files:
        with open(filename, "r") as f:
            for line in f:
                # Diagnosis lines start with # Dx or #Dx
                if line.startswith("# Dx") or line.startswith("#Dx"):
                    codes = line.split(": ")[1].split(",")
                    for c in codes:
                        classes.add(c.strip())
    return sorted(classes)


# Identify patients with arrhythmias and detect those with multiple diagnoses
def check_arrhythmias(headers, arrhythmias):
    arr_codes = list(arrhythmias.values())
    multiple_arrhythmia_patients = []
    patients_with_arrhythmia = []
    multiple_codes = []

    arrhythmias_inv = {v: k for k, v in arrhythmias.items()}

    for idx, header in enumerate(headers):
        for line in header:
            if line.startswith("# Dx:") or line.startswith("#Dx"):
                codes = line.split(": ")[1].strip().split(",")
                # Keep only codes present in the arrhythmias dictionary
                found = [arrhythmias_inv[c] for c in codes if c in arr_codes]
                if found:
                    patients_with_arrhythmia.append(idx)
                # If more than one valid code is present, mark as multiple arrhythmia
                if len(found) > 1:
                    multiple_arrhythmia_patients.append(idx)
                    multiple_codes.append(found)

    return patients_with_arrhythmia, multiple_arrhythmia_patients, multiple_codes


# Check whether age and sex fields are properly formatted
def check_age_sex(headers):
    invalid_age_patients, invalid_sex_patients = [], []

    for idx, header in enumerate(headers):
        for line in header:
            if line.startswith("# Age:") or line.startswith("#Age:"):
                age = line.split(": ")[1].strip()
                # Age must be numeric
                if not age.isdigit():
                    invalid_age_patients.append(idx)
            if line.startswith("# Sex:") or line.startswith("#Sex:"):
                sex = line.split(": ")[1].strip().lower()
                # Only male/female are accepted
                if sex not in ["male", "female"]:
                    invalid_sex_patients.append(idx)

    return invalid_age_patients, invalid_sex_patients


# Verify that arrhythmia patients also have a valid age
def check_arrhythmia_with_age(headers, arrhythmias):
    arr_codes = list(arrhythmias.values())
    patients_missing_age = []

    for idx, header in enumerate(headers):
        has_arrhythmia, has_age = False, True
        for line in header:
            if line.startswith("# Dx:") or line.startswith("#Dx"):
                codes = line.split(": ")[1].strip().split(",")
                # Patient is considered arrhythmic if any code matches
                if any(code in arr_codes for code in codes):
                    has_arrhythmia = True
            if line.startswith("# Age:") or line.startswith("#Age:"):
                age = line.split(": ")[1].strip()
                if not age.isdigit():
                    has_age = False
        # Keep patients with arrhythmia but missing valid age
        if has_arrhythmia and not has_age:
            patients_missing_age.append(idx)

    return patients_missing_age


# Copy valid patient records to the output directory
def filter_and_copy(headers, patients_with_arrhythmia, multiple_arrhythmia_patients,
                    patients_missing_age, output_dir, input_dir):
    hea_files = []
    # Collect all header files keeping subfolder order
    for subdir in sorted(os.listdir(input_dir)):
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.isdir(subdir_path):
            for f in os.listdir(subdir_path):
                if f.lower().endswith(".hea"):
                    hea_files.append(os.path.join(subdir_path, f))

    # Evaluate conditions for each patient
    for idx, header in enumerate(headers):
        has_multiple = idx in multiple_arrhythmia_patients
        has_no_age = idx in patients_missing_age
        has_arrhythmia = idx in patients_with_arrhythmia

        sep = header[0].split()
        samples, freq = int(sep[3]), int(sep[2])
        duration = samples / freq
        wrong_duration = duration != 10

        # Copy only patients that meet all requirements
        if has_arrhythmia and not has_multiple and not has_no_age and not wrong_duration:
            hea_file = hea_files[idx]
            mat_file = hea_file.replace(".hea", ".mat")

            if os.path.exists(mat_file):
                shutil.copy(hea_file, output_dir)
                shutil.copy(mat_file, output_dir)