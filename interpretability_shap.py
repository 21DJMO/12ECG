# Libraries
import numpy as np
import shap
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Load dataset from .h5 file
h5_file_path = "/mnt/Home-Group/kruiz_cps/12ECG/full_dataset_12ECG.h5"
with h5py.File(h5_file_path, "r") as h5_file:
    x_data = np.array(h5_file["Signals"])[:, :, :12]   # Use 12 leads
    y_data = np.array(h5_file["Labels"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.15, random_state=42, stratify=y_data.argmax(axis=1)
)

# Load best unimodal model trained on 12 leads
best_model = load_model("model_12leads/model_fold7.keras")

# Load training indices from text file
# These correspond to the training set of the best fold in 12-lead cross-validation (unimodal model)
with open("idx_train.txt", "r") as file:
    content = file.read()
    train_indices = [int(num) for num in content.split()]

print(len(train_indices))  # Print number of selected samples

# Subset training data using the selected indices
X_train_selected = X_train[train_indices]

# Define ECG leads and arrhythmia classes
leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
arrhythmias = {
    "AF":     "164889003",  # Atrial fibrillation
    "AFL":    "164890007",  # Atrial flutter
    "SA":     "427393009",  # Sinus arrhythmia
    "SB":     "426177001",  # Sinus bradycardia
    "STach":  "427084000",  # Sinus tachycardia
    "NSR":    "426783006"   # Normal sinus rhythm
}
class_names = list(arrhythmias.keys())

# Select background and test samples for SHAP analysis
background = X_train_selected[np.random.choice(X_train_selected.shape[0], 100, replace=False)]
test_samples = X_train_selected[np.random.choice(X_train_selected.shape[0], 50, replace=False)]

# Initialize SHAP explainer for the unimodal model and compute SHAP values
explainer = shap.DeepExplainer(best_model, background)
shap_values = explainer.shap_values(test_samples)  # Shape: (50, 2500, 12, 6)

# Initialize container for average importance per lead
lead_importance = np.zeros((len(test_samples), len(leads)))

# Loop through each class and accumulate SHAP contributions
for class_idx in range(len(class_names)):
    sv = shap_values[..., class_idx]              # Shape: (50, 2500, 12)
    sv_class_avg = np.abs(sv).mean(axis=1)        # Average over time dimension
    lead_importance += sv_class_avg               # Accumulate across classes

# Normalize by number of classes to obtain weighted average per lead
lead_importance /= len(class_names)

# Plot SHAP importance values as bar chart
plt.figure(figsize=(10, 5))
shap.summary_plot(
    lead_importance,
    features=test_samples.mean(axis=1),
    feature_names=leads,
    plot_type="bar",
    color="#F70069",
    show=False
)
plt.tight_layout()
plt.savefig("shap_lead_importance.pdf", bbox_inches="tight")
plt.close()