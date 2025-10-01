# Libraries
import numpy as np, os
from scipy.io import loadmat
import h5py
from keras.utils import to_categorical


# Load all patient data from a folder
def train_12ECG_classifier(input_directory):
	header_files = []
	# Find all .hea files (header files) in directory and subdirectories
	for root, _, files in os.walk(input_directory):
		for f in files:
			g = os.path.join(root, f)
			if f.lower().endswith('.hea') and os.path.isfile(g):
				header_files.append(g)

	recordings, headers = [], []
	# For each patient, load both ECG signal and header information
	for header_file in header_files:
		recording, header = load_challenge_data(header_file)
		recordings.append(recording)
		headers.append(header)

	return recordings, headers, len(header_files)


# Load ECG signal and header metadata from one patient
def load_challenge_data(header_file):
	# Read metadata from .hea file
	with open(header_file, 'r') as f:
		header = f.readlines()

	# Load corresponding .mat file with ECG signal
	mat_file = header_file.replace('.hea', '.mat')
	x = loadmat(mat_file)

	# Extract signal values as numpy array
	recording = np.asarray(x['val'], dtype=np.float64)
	return recording, header


# Extract demographic information (age, sex)
def extract_demographic(headers):
	ages, sexes = [], []
	for h in headers:
		# Age is stored in line 14 of header
		age = int(h[13].split(":")[1].strip())

		# Sex is stored in line 15 of header
		sex_str = h[14].split(":")[1].strip()
		sex_value = 0 if sex_str == "Female" else 1

		ages.append(age)
		sexes.append(sex_value)

	# Return array with shape [patients, 2]
	return np.array([ages, sexes]).T


# Format ECG signals into [patients, samples, leads]
def extract_signals(recordings):
	recordings_array = np.array(recordings)
	# Transpose to have samples in second dimension
	return np.transpose(recordings_array, (0, 2, 1))


# Extract arrhythmia labels as one-hot vectors
def extract_labels(headers):
	# Dictionary of arrhythmia codes of interest
	arrhythmias = {
		"AF": "164889003",
		"AFL": "164890007",
		"SA": "427393009",
		"SB": "426177001",
		"STach": "427084000",
		"NSR": "426783006"
	}

	code_to_index = {v: i for i, v in enumerate(arrhythmias.values())}
	y_indices = []

	for h in headers:
		found = False
		for line in h:
			# Diagnosis lines start with # Dx or #Dx
			if line.startswith('# Dx:') or line.startswith('#Dx'):
				# Multiple codes can appear separated by commas
				codes = line.split(': ')[1].strip().split(',')
				for code in codes:
					code = code.strip()
					# Only codes included in arrhythmias dictionary are kept
					if code in code_to_index:
						# If multiple valid codes appear, the first one is selected
						y_indices.append(code_to_index[code])
						found = True
						break
			if found:
				break
		if not found:
			raise ValueError("No valid diagnosis code found in this record")

	# Convert integer labels to one-hot encoded format
	return to_categorical(np.array(y_indices), num_classes=6)


# Input directory
input_directory = "/mnt/Home-Group/kruiz_cps/Dataset_12ECG/DATABASESUB"

# Load recordings and headers
recordings, headers, num_files = train_12ECG_classifier(input_directory)

# Extract demographic, signals, and labels
demographic = extract_demographic(headers)
x_data = extract_signals(recordings)
y_data = extract_labels(headers)

# Print array shapes for quick verification
print(x_data.shape, demographic.shape, y_data.shape)

# Save dataset in HDF5 format
with h5py.File("dataset_12ECG.h5", "w") as f:
	f.create_dataset("Signals", data=x_data)
	f.create_dataset("Demographic", data=demographic)
	f.create_dataset("Labels", data=y_data)