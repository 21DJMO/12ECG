# Libraries
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pandas as pd

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# ECG lead configurations
lead_configs = {
    '2leads': np.array([0, 4]),
    '4leads': np.array([0, 1, 4, 5]),
    '6leads': np.array([0, 1, 3, 4, 5, 6]),
    '12leads': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
}

# Select lead configuration
selected_leads = '2leads'    # Options: '2leads', '4leads', '6leads', '12leads'
indices = lead_configs[selected_leads]
num_lead = len(indices)

# Load dataset
h5_file_path = "/mnt/Home-Group/kruiz_cps/12ECG/full_dataset_12ECG.h5"
with h5py.File(h5_file_path, "r") as h5_file:
    x_data_ecg = np.array(h5_file["Signals"])[:, :, indices]
    x_data_features = np.array(h5_file["Features"])
    y_data = np.array(h5_file["Labels"])

# Train/test split (same as training)
X_train_ecg, X_test_ecg, X_train_features, X_test_features, y_train, y_test = train_test_split(
    x_data_ecg, x_data_features, y_data, test_size=0.15, random_state=42, stratify=y_data.argmax(axis=1)
)

# Normalize ECG signals (z-score based on training set)
X_train_ecg_reshaped = X_train_ecg.reshape(-1, num_lead)
mean_ecg = np.mean(X_train_ecg_reshaped, axis=0, keepdims=True)
std_ecg = np.std(X_train_ecg_reshaped, axis=0, keepdims=True)
X_test_ecg_reshaped = X_test_ecg.reshape(-1, num_lead)
X_test_ecg = ((X_test_ecg_reshaped - mean_ecg) / std_ecg).reshape(X_test_ecg.shape)

# Normalize tabular features
scaler_features = StandardScaler()
X_train_features = scaler_features.fit_transform(X_train_features)
X_test_features = scaler_features.transform(X_test_features)

# Free memory
del x_data_ecg, x_data_features, y_data, X_train_ecg_reshaped, X_test_ecg_reshaped

# Load best trained model
model_path = "/mnt/Home-Group/kruiz_cps/12ECG/multimodal_thesis_v1/multimodal_2leads/multimodal_2leads_fold1.keras"
best_model = keras.models.load_model(model_path)

# Challenge metric configuration
classes = [164889003, 164890007, 427393009, 426177001, 427084000, 426783006]  # AF, AFL, SA, SB, STach, NSR
normal_class = 426783006
weights_file = "/mnt/Home-Group/kruiz_cps/12ECG/weights.csv"

# Load weight matrix used for the challenge metric
def load_weights(weights_file, classes):
    df = pd.read_csv(weights_file, index_col=0)
    df.columns = df.columns.astype(int)
    df.index = df.index.astype(int)
    return df.loc[classes, classes].to_numpy()

# Compute a simple confusion matrix with predicted vs true classes
def compute_modified_confusion_matrix(labels, outputs):
    num_recordings, num_classes = labels.shape
    A = np.zeros((num_classes, num_classes))
    for i in range(num_recordings):
        true_class = np.argmax(labels[i])
        pred_class = np.argmax(outputs[i])
        A[true_class, pred_class] += 1
    return A

# Compute normalized challenge score
def compute_challenge_metric(weights, labels, outputs, classes, normal_class):
    num_recordings, num_classes = labels.shape
    normal_index = classes.index(normal_class)

    # Observed score
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Correct score (ideal predictions)
    correct_outputs = labels
    A = compute_modified_confusion_matrix(correct_outputs, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Inactive score (all predicted as normal)
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.int32)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    # Normalization
    if correct_score != inactive_score:
        normalized_score = (observed_score - inactive_score) / (correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score

# Evaluate the model on the test set and compute standard + challenge metrics
def evaluate_model(X_ecg, X_features, y_true, model, mode):
    print(f"\nEvaluation ({mode}):")
    eval = model.evaluate([X_ecg, X_features], y_true, verbose=0)
    print("Test loss:", eval[0])
    print("Test accuracy:", eval[1])

    # Predictions
    y_pred_prob = model.predict([X_ecg, X_features], verbose=0)
    y_pred = np.zeros_like(y_pred_prob)
    y_pred[np.arange(len(y_pred_prob)), y_pred_prob.argmax(axis=1)] = 1

    # Standard metrics
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    print("Precision (macro):", precision)
    print("Recall (macro):", recall)
    print("F1-score (macro):", f1)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["AF", "AFL", "SA", "SB", "STach", "NSR"]))

    # Challenge metric
    weights = load_weights(weights_file, classes)
    normalized_score = compute_challenge_metric(weights, y_true, y_pred, classes, normal_class)

    print("\nChallenge score (normalized):", normalized_score)

# Run evaluation
evaluate_model(X_test_ecg, X_test_features, y_test, best_model, f"multimodal_{selected_leads}")