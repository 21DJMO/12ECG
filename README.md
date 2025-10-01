# Cardiac Arrhythmias Detection with 12-Lead ECG Using a Multimodal Neural Network

Karla Ruíz, Didier J. Moreno, Harold H. Rodriguez, Camilo Santos, Carlos A. Fajardo

This repository contains the code for the project focused on detecting cardiac arrhythmias using a multimodal neural network with 12-lead ECG data, while incorporating interpretability to identify the most influential leads.

## Repository Structure
This repository includes Python scripts for data processing, model training, testing, and interpretability. Each file is described below:

- **filter_ecg_dataset.py**: Cleans the raw dataset by excluding records without age/sex, with multiple arrhythmias, incorrect duration, or outside the six selected target classes.
- **downsample_ecg_dataset.py**: Reduces ECG sampling rate from 500 Hz to 250 Hz.
- **load_dataset.py**: Builds structured HDF5 files containing ECG signals, class labels, and demographic features for efficient training and evaluation.
- **extract_features.py**: Extracts time-domain Heart Rate Variability (HRV) features using NeuroKit2, and integrates demographic variables.  
- **train_unimodal.py**: Trains a convolutional neural network (CNN) using ECG signals only, with configurable setups of 12, 6, 4, or 2 leads.  
- **train_multimodal.py**: Trains a multimodal model that combines ECG signals with HRV-based and demographic features.  
- **test_unimodal.py**: Evaluates unimodal models on the test set, reporting metrics such as accuracy, precision, recall, F1-score, and the PhysioNet Challenge score.  
- **test_multimodal.py**: Evaluates multimodal models on the test set, following the same metrics as the unimodal evaluation.
- **interpretability_shap.py**: Computes Shapley Additive Explanations (SHAP) values on the best unimodal 12-lead model to rank lead importance and analyze interpretability

## Usage
1. Download the PhysioNet Challenge 2020 dataset from https://physionet.org/content/challenge-2020/1.0.2/ and the 2021 dataset from https://physionet.org/content/challenge-2021/1.0.3/.
2. Place the downloaded datasets in the locations specified in the scripts (e.g., root directory or configured paths).
3. Run preprocessing scripts in order:
   ```
   python filter_ecg_dataset.py
   python downsample_ecg_dataset.py
   python load_dataset.py
   python extract_features.py
   ```
4. Train models:
   - Unimodal: `python train_unimodal.py`
   - Multimodal: `python train_multimodal.py`
5. Apply interpretability: `python interpretability_shap.py` (on the trained 12-lead unimodal model).
6. Test models:
   - Unimodal: `python test_unimodal.py`
   - Multimodal: `python test_multimodal.py`

Adjust hyperparameters (e.g., number of leads) in the scripts as needed.
