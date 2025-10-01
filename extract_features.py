# Libraries
import h5py
import neurokit2 as nk
import pandas as pd
import numpy as np
import warnings

# Input dataset
h5_file_path = "/mnt/Home-Group/kruiz_cps/12ECG/dataset_12ECG.h5"

# Output file names
excel_file_name = "ecg_features.xlsx"
new_h5_file_name = "full_dataset_12ECG.h5"

# Open original dataset in read mode
with h5py.File(h5_file_path, "r") as f:
	# Load signals, labels, and demographic information
	ecg_signals = f["Signals"][:]
	labels = f["Labels"][:]
	demographic = f["Demographic"][:]

print("Original dataset loaded:", ecg_signals.shape, labels.shape, demographic.shape)

# Extract one ECG lead (lead II, index 1 in 0-based)
test_lead = ecg_signals[:, :, 1]
print("Shape of selected lead:", test_lead.shape)

# Ignore warnings to keep output clean
warnings.filterwarnings("ignore")

# Helper function to unwrap values stored in nested containers
def unwrap(cell):
	while isinstance(cell, (list, np.ndarray, pd.Series)) and len(cell) > 0:
		cell = cell[0]
	return cell

# Features to be extracted
time_features = [
	"ECG_Rate_Mean",
	"HRV_MeanNN",
	"HRV_RMSSD",
	"HRV_SDSD",
	"HRV_MedianNN",
	"HRV_MinNN",
	"HRV_MaxNN",
	"HRV_pNN20",
	"HRV_SD1"
]
extra = ["Age", "Genre"]
desired_columns = time_features + extra

# Lists for storing results and errors
all_features = []
problematic_samples = []
all_indices = []

# Processing parameters
sampling_rate = 250
min_length = 500
min_std = 0.01

# Process each sample
for i in range(len(test_lead)):
	sample_info = {
		"index": i,
		"length": len(test_lead[i]),
		"std": np.std(test_lead[i]),
		"min": np.min(test_lead[i]),
		"max": np.max(test_lead[i]),
		"has_nan": np.isnan(test_lead[i]).any(),
		"has_inf": np.isinf(test_lead[i]).any(),
		"r_peaks_count": 0,
		"error": None
	}
	all_indices.append(i)

	# Initial checks: too short, flat, or invalid values
	if sample_info["length"] < min_length:
		sample_info["error"] = f"Too short ({sample_info['length']} samples)"
	elif sample_info["std"] < min_std:
		sample_info["error"] = f"Low variability (std={sample_info['std']})"
	elif sample_info["has_nan"] or sample_info["has_inf"]:
		sample_info["error"] = "Contains NaN or Inf values"

	if sample_info["error"]:
		problematic_samples.append(sample_info)
		print(f"Sample {i}: {sample_info['error']} → storing NaN")
		features = pd.DataFrame({feat: [np.nan] for feat in time_features}, index=[i])
		all_features.append(features)
		continue

	try:
		# Clean signal and extract R-peaks
		cleaned_ecg = nk.ecg_clean(test_lead[i], sampling_rate=sampling_rate, method="biosppy")
		signals, info = nk.ecg_process(cleaned_ecg, sampling_rate=sampling_rate, method="pantompkins")
		sample_info["r_peaks_count"] = len(info["ECG_R_Peaks"])

		# Must detect at least 2 R-peaks
		if sample_info["r_peaks_count"] < 2:
			sample_info["error"] = f"Too few R-peaks ({sample_info['r_peaks_count']})"
			problematic_samples.append(sample_info)
			print(f"Sample {i}: {sample_info['error']} → storing NaN")
			features = pd.DataFrame({feat: [np.nan] for feat in time_features}, index=[i])
			all_features.append(features)
			continue

		# Extract HRV features
		features = nk.ecg_analyze(signals, sampling_rate=sampling_rate, method="auto")
		features = features[[col for col in time_features if col in features.columns]]
		features = features.apply(lambda x: x.apply(unwrap) if x.name in features.columns else x)
		features.index = [i]
		all_features.append(features)

	except Exception as e:
		sample_info["error"] = f"Processing error: {str(e)}"
		problematic_samples.append(sample_info)
		print(f"Sample {i}: {sample_info['error']} → storing NaN")
		features = pd.DataFrame({feat: [np.nan] for feat in time_features}, index=[i])
		all_features.append(features)
		continue

# DataFrame with problematic cases
problematic_df = pd.DataFrame(problematic_samples)
if not problematic_df.empty:
	print("\nProblematic samples found:")
	print(problematic_df)
else:
	print("\nNo problematic samples found.")

# Build final features array
if all_features:
	all_features_df = pd.concat(all_features, ignore_index=False)

	# Ensure all expected columns are present
	for col in time_features:
		if col not in all_features_df.columns:
			all_features_df[col] = np.nan
	all_features_df = all_features_df[time_features]

	# Replace NaN/Inf with zeros in time features
	all_features_df[time_features] = all_features_df[time_features].replace([np.inf, -np.inf], np.nan).fillna(0)

	# Add demographic info
	try:
		all_features_df["Age"] = [demographic[j, 0] for j in all_indices]
		all_features_df["Genre"] = [demographic[j, 1] for j in all_indices]
	except Exception as e:
		print(f"Error adding demographic data: {str(e)}")

	# Reindex to keep consistent patient order
	all_features_df = all_features_df.reindex(range(len(test_lead)))

	# Save features to Excel (useful to review all patient features manually)
	all_features_df.to_excel(excel_file_name, index=True)
	print(f"\nExcel file saved: {excel_file_name}")

	# Convert DataFrame to numpy array for HDF5
	features_array = all_features_df.to_numpy()
	print("\nFinal features shape:", features_array.shape)

	# Save new full dataset in HDF5 format
	with h5py.File(new_h5_file_name, "w") as new_h5_file:
		new_h5_file.create_dataset("Signals", data=ecg_signals)
		new_h5_file.create_dataset("Features", data=features_array)
		new_h5_file.create_dataset("Labels", data=labels)

	print(f"\nFull dataset saved to {new_h5_file_name}")

else:
	print("\nNo valid features generated. All samples failed.")