# Libraries
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		# Select a specific GPU by index
		tf.config.set_visible_devices(gpus[2], 'GPU')
		# Allow memory growth on the selected GPU
		tf.config.experimental.set_memory_growth(gpus[2], True)
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
selected_leads = '12leads'    # Options: '2leads', '4leads', '6leads', '12leads'
indices = lead_configs[selected_leads]
num_lead = len(indices)

# Load dataset with ECG signals, tabular features, and labels
h5_file_path = "/mnt/Home-Group/kruiz_cps/12ECG/full_dataset_12ECG.h5"
with h5py.File(h5_file_path, "r") as h5_file:
	x_data = np.array(h5_file["Signals"])[:, :, indices]
	feat_data = np.array(h5_file["Features"])
	y_data = np.array(h5_file["Labels"])

# Split into training and testing sets
X_train, X_test, feat_train, feat_test, y_train, y_test = train_test_split(
	x_data, feat_data, y_data, test_size=0.15, random_state=42, stratify=y_data.argmax(axis=1)
)

# Normalize ECG signals
X_train_reshaped = X_train.reshape(-1, num_lead)
mean = np.mean(X_train_reshaped, axis=0, keepdims=True)
std = np.std(X_train_reshaped, axis=0, keepdims=True)
X_train = ((X_train_reshaped - mean) / std).reshape(X_train.shape)
X_test = ((X_test.reshape(-1, num_lead) - mean) / std).reshape(X_test.shape)

# Normalize tabular features
scaler = StandardScaler()
feat_train = scaler.fit_transform(feat_train)
feat_test = scaler.transform(feat_test)

# Free memory from unused variables
del x_data, feat_data, y_data, X_train_reshaped


# Build multimodal model
def build_model():
	# ECG branch
	input_ecg = keras.Input(shape=(2500, num_lead))
	y = layers.Conv1D(128, 5, strides=3, activation='relu', padding='same',
					  kernel_initializer='he_uniform', kernel_regularizer='l2')(input_ecg)
	y = layers.BatchNormalization()(y)
	y = layers.MaxPool1D(pool_size=2, strides=3)(y)

	y = layers.Conv1D(32, 7, strides=1, activation='relu', padding='same',
					  kernel_initializer='he_uniform', kernel_regularizer='l2')(y)
	y = layers.BatchNormalization()(y)
	y = layers.MaxPool1D(pool_size=2, strides=2)(y)
	y = layers.Dropout(0.3)(y)

	y = layers.Conv1D(32, 11, strides=1, activation='relu', padding='same',
					  kernel_initializer='he_uniform', kernel_regularizer='l2')(y)
	y = layers.BatchNormalization()(y)
	y = layers.MaxPool1D(pool_size=2, strides=2)(y)
	y = layers.Dropout(0.3)(y)

	y = layers.Conv1D(128, 11, strides=2, activation='relu', padding='same',
					  kernel_initializer='he_uniform', kernel_regularizer='l2')(y)
	y = layers.BatchNormalization()(y)
	y = layers.MaxPool1D(pool_size=2, strides=2)(y)
	y = layers.Dropout(0.3)(y)

	y = layers.Conv1D(256, 11, strides=1, activation='relu', padding='same',
					  kernel_initializer='he_uniform', kernel_regularizer='l2')(y)
	y = layers.BatchNormalization()(y)
	y = layers.MaxPool1D(pool_size=2, strides=2)(y)
	y = layers.Dropout(0.3)(y)

	y = layers.Conv1D(512, 11, strides=1, activation='relu', padding='same',
					  kernel_initializer='he_uniform', kernel_regularizer='l2')(y)
	y = layers.BatchNormalization()(y)
	y = layers.MaxPool1D(pool_size=2, strides=2)(y)
	y = layers.Dropout(0.3)(y)

	y = layers.Conv1D(128, 11, strides=3, activation='relu', padding='same',
					  kernel_initializer='he_uniform', kernel_regularizer='l2')(y)
	y = layers.BatchNormalization()(y)
	y = layers.MaxPool1D(pool_size=2, strides=2)(y)
	y = layers.Dropout(0.3)(y)

	y = layers.Flatten()(y)

	# Tabular features branch
	input_feat = keras.Input(shape=(11,))
	z = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(input_feat)
	z = layers.Dropout(0.3)(z)
	z = layers.Dense(32, activation='relu', kernel_initializer='he_uniform')(z)
	z = layers.Dropout(0.3)(z)

	# Merge branches
	combined = layers.Concatenate()([y, z])
	combined = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(combined)
	combined = layers.Dropout(0.5)(combined)

	# Output layer
	output = layers.Dense(6, activation='softmax', kernel_initializer='glorot_normal')(combined)

	model = keras.Model(inputs=[input_ecg, input_feat], outputs=output)
	return model


# Initialize metrics and results tracking
accuracies, precisions, recalls, f1s = [], [], [], []
all_preds, all_true = [], []
histories = []

# Cross-validation setup (10 folds, stratified)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Results file for the selected lead configuration
results_file = f"/mnt/Home-Group/kruiz_cps/12ECG/multimodal_{selected_leads}_v2/training_results_{selected_leads}_v2.txt"
with open(results_file, "w") as f:
	f.write(f"Training results ({selected_leads})\n\n")

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train.argmax(axis=1))):
	print(f"\nFold {fold + 1}")
	
	# Split current fold into train/validation
	X_train_k, X_val = X_train[train_idx], X_train[val_idx]
	feat_train_k, feat_val = feat_train[train_idx], feat_train[val_idx]
	y_train_k, y_val = y_train[train_idx], y_train[val_idx]

	# Build and compile model
	model = build_model()
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Define training callbacks
	checkpoint = callbacks.ModelCheckpoint(
		f'multimodal_{selected_leads}_fold{fold+1}.keras', monitor='val_loss',
		save_best_only=True, verbose=1
	)
	early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
	reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)

	# Train model on current fold
	history = model.fit(
		[X_train_k, feat_train_k], y_train_k,
		validation_data=([X_val, feat_val], y_val),
		epochs=5000, batch_size=256,
		callbacks=[checkpoint, early_stopping, reduce_lr],
		verbose=0
	)
	histories.append(history.history)

	# Save learning curve plot
	train_loss = history.history['loss']
	val_loss = history.history['val_loss']
	plt.figure(figsize=(6, 4))
	plt.plot(train_loss, label='Train')
	plt.plot(val_loss, label='Validation')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title(f'Learning Curves of the Multimodal Model for {num_lead} Leads')
	plt.legend()
	plt.grid()
	plt.savefig(
		f"/mnt/Home-Group/kruiz_cps/12ECG/multimodal_{selected_leads}_v2/loss_curve_multimodal_{selected_leads}_fold{fold+1}.pdf",
		dpi=300
	)
	plt.close()

	# Evaluate best model for this fold
	best_model = keras.models.load_model(f'multimodal_{selected_leads}_fold{fold+1}.keras')
	y_pred = best_model.predict([X_val, feat_val], verbose=0).argmax(axis=1)
	y_true = y_val.argmax(axis=1)

	# Store predictions and ground truth
	all_preds.extend(y_pred)
	all_true.extend(y_true)

	# Compute evaluation metrics
	acc = accuracy_score(y_true, y_pred)
	prec = precision_score(y_true, y_pred, average='macro')
	rec = recall_score(y_true, y_pred, average='macro')
	f1 = f1_score(y_true, y_pred, average='macro')

	# Save metrics into their respective lists
	accuracies.append(acc)
	precisions.append(prec)
	recalls.append(rec)
	f1s.append(f1)

	# Print and log fold results
	print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
	with open(results_file, "a") as f:
		f.write(f"Fold {fold + 1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}\n")

# Compute final metrics across folds
def resumen(name, values):
	mean = np.mean(values)
	std = np.std(values)
	print(f"{name}: {mean:.4f} ± {std:.4f}")
	return mean, std

print("\nFinal summary of the 10 folds:")
with open(results_file, "a") as f:
	f.write("\nFinal summary of the 10 folds:\n")
	for name, values in zip(
		["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"],
		[accuracies, precisions, recalls, f1s]
	):
		mean, std = resumen(name, values)
		f.write(f"{name}: {mean:.4f} ± {std:.4f}\n")

# Compute confusion matrix across all folds
print("\nGlobal confusion matrix (all folds):")
global_conf_matrix = confusion_matrix(all_true, all_preds)
print(global_conf_matrix)
with open(results_file, "a") as f:
	f.write("\nGlobal confusion matrix (all folds):\n")
	f.write(str(global_conf_matrix) + "\n")