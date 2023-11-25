import os
import joblib
import numpy as np
import pandas as pd
import wfdb
import pywt
import tensorflow as tf
from scipy.signal import iirnotch, lfilter, find_peaks

# Constantes
MODEL_RHYTHM_FILE_PATH = '/Users/ivanlorenzanabelli/Arrhythmia-Detector/Models/modelo_ecg_rhythm.h5'
MODEL_BEAT_FILE_PATH = '/Users/ivanlorenzanabelli/Arrhythmia-Detector/Models/modelo_ecg_beat.h5'
SCALER_FILE_PATH = '/Users/ivanlorenzanabelli/Arrhythmia-Detector/Models/scaler_ecg.pk1'

# Function definitions
def get_ml_ii_index(header_lines):
    """
    The function `get_ml_ii_index` returns the index of the line before the line containing 'MLII' in a
    list of header lines.
    
    :param header_lines: A list of strings representing the header lines of a file
    :return: the index of the line before the line that contains 'MLII' in the header_lines list. If no
    line contains 'MLII', it returns None.
    """
    for i, line in enumerate(header_lines):
        if 'MLII' in line:
            return i - 1  # Subtract 1 to compensate for the first header line
    return None  # If MLII is not found, return None

def calculate_fft_and_wavelet(signal_windowed, fs):
    """
    The function `calculate_fft_and_wavelet` calculates the FFT power, dominant frequency, wavelet
    energy, and total Shannon entropy of a given signal windowed with a specific window function.
    
    :param signal_windowed: The signal_windowed parameter is a windowed segment of the signal that you
    want to analyze. It should be a 1-dimensional array of numerical values representing the amplitude
    of the signal at each time point
    :param fs: fs is the sampling frequency of the signal. It represents the number of samples taken per
    second
    :return: The function `calculate_fft_and_wavelet` returns four values: `total_power`,
    `dominant_freq`, `wavelet_Energy`, and `total_shannon_entropy`.
    """
    ## FFT y cálculo de energía
    fft_coeffs = np.fft.rfft(signal_windowed)
    fft_power = np.abs(fft_coeffs)**2

    # Densidad espectral de potencia (PSD)
    N = len(signal_windowed)
    df = fs / N
    PSD = (np.abs(fft_coeffs) ** 2) / df
    PSD[1:-1] *= 2  # Ajuste para frecuencias no DC
    total_PSD = np.sum(PSD)

    # Energía espectral total
    total_spectral_energy = np.sum(fft_power)

    # Transformada Wavelet
    scales = np.arange(1, 17)  # Ajustar según las necesidades
    coefc, _ = pywt.cwt(signal_windowed, scales, 'mexh')
    wavelet_energy = np.sum(coefc**2)

    # Entropía de Shannon
    energy = np.sum(coefc**2, axis=1)
    energy[energy == 0] = np.finfo(float).eps  # Evitar división por cero
    prob_dist = coefc**2 / energy[:, None]
    prob_dist /= np.sum(prob_dist, axis=1)[:, None]
    shannon_entropy = -np.sum(prob_dist * np.log2(prob_dist), axis=1)
    total_shannon_entropy = np.mean(shannon_entropy)

    return total_spectral_energy, total_PSD, wavelet_energy, total_shannon_entropy

def apply_window(signal, peak, fs, width=1):
    """
    The function `apply_window` extracts a window of a specified width around a peak in a signal, given
    the signal, the peak index, the sampling frequency, and the width of the window.
    
    :param signal: The input signal that you want to apply the window to
    :param peak: The peak parameter represents the index of the peak R in the signal. It is used to
    determine the start and end indices of the window
    :param fs: The parameter "fs" represents the sampling frequency of the signal. It is the number of
    samples taken per second
    :param width: The width parameter represents the duration of the window in seconds. It determines
    the length of the segment of interest around the peak, defaults to 1 (optional)
    :return: the windowed segment of the signal based on the specified peak and width.
    """
    start_index = max(peak - int(width/2 * fs), 0)
    end_index = min(peak + int(width/2 * fs), len(signal))

    if start_index < end_index:
        return signal[start_index:end_index]
    return None

def min_max_normalize(signal):
    """
    The function `min_max_normalize` takes a signal as input and returns a normalized version of the
    signal using min-max normalization.
    
    :param signal: The input signal is a one-dimensional array or list of numerical values
    :return: the normalized signal.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(signal)
    normalized_signal = (signal - min_val) / range_val
    return normalized_signal

def extract_features_from_window(signal_windowed, fs):
    """
    The function `extract_features_from_window` takes a windowed signal and its sampling frequency as
    input, and extracts various features from the signal window.
    
    :param signal_windowed: The signal_windowed parameter is a windowed segment of a signal. It is
    assumed to be centered around an R peak in an electrocardiogram (ECG) signal
    :param fs: fs is the sampling frequency of the signal. It represents the number of samples taken per
    second
    :return: an array of features extracted from the input signal window.
    """
    signal_windowed_normalized = min_max_normalize(signal_windowed)
    # Asumimos que signal_windowed ya es una ventana centrada en un pico R
    distanceW = int(0.45 * fs)
    peaksW, _ = find_peaks(signal_windowed_normalized, distance=distanceW)
    peaksW = peaksW[signal_windowed_normalized[peaksW] > 0.6]
    std_val = np.std(signal_windowed_normalized)

    # Cálculo de FFT y Wavelet
    total_spectral_energy, total_psd, wavelet_energy, total_shannon_entropy = calculate_fft_and_wavelet(signal_windowed_normalized, fs)

    # Extraer características
    features = [
        len(peaksW),
        total_spectral_energy,
        total_psd,
        wavelet_energy,
        total_shannon_entropy,
        std_val
    ]

    return np.array(features)

def predict_ecg(features):
    """
    The `predict_ecg` function takes in a list of 6 ECG features, loads pre-trained models and a scaler,
    normalizes the features, and makes predictions for beat and rhythm labels.
    
    :param features: The 'features' parameter is a list or array containing 6 numerical values. Each
    value represents a specific feature of an electrocardiogram (ECG) signal. The features are:
    :return: the predicted beat label and rhythm label.
    """
    feature_labels = ["RPeakCount", "SpectralEnergy", "TotalPSD", "WaveletEnergy", "ShannonEntropy", "SignalSTD"]
    
    # Asegúrate de que 'features' es un array o lista con 6 elementos
    if len(features) != 6:
        return "Error: Se requieren exactamente 6 características"

    # Convertir las características en un DataFrame
    df = pd.DataFrame([features], columns=feature_labels)

    # Cargar el modelo y el scaler
    modelRhythm = tf.keras.models.load_model(MODEL_RHYTHM_FILE_PATH)
    modelBeat = tf.keras.models.load_model(MODEL_BEAT_FILE_PATH)
    scaler = joblib.load(SCALER_FILE_PATH)  # Asegúrate de haber guardado tu scaler

    # Normalizar las características
    features_scaled = scaler.transform(df)

    # Hacer predicciones
    predictionsBeats = modelBeat.predict(features_scaled)
    predictionsRhythm = modelRhythm.predict(features_scaled)
    
    # Convertir las predicciones a etiquetas legibles (opcional)
    beat_label = np.argmax(predictionsBeats, axis=1)
    rhythm_label = np.argmax(predictionsRhythm, axis=1)

    rhythm_labelP = get_rhythm_label(rhythm_label)
    beat_labelP = get_beat_label(beat_label)

    return beat_labelP, rhythm_labelP

def get_beat_label(prediction):
    """
    The function `get_beat_label` takes a prediction value and returns the corresponding beat label
    based on a predefined mapping.
    
    :param prediction: The `prediction` parameter is a list containing the predicted value for a beat
    :return: The function `get_beat_label` returns the label corresponding to the given prediction
    value. If the prediction value is found in the `annotation_mapping` dictionary, the corresponding
    label is returned. If the prediction value is not found in the dictionary, the function returns
    "Unknown".
    """
    pred_value = prediction[0]
    annotation_mapping = {
        0: "Normal",
        "L": "Left bundle branch block beat",
        2: "Right bundle branch block beat",
        5: "Atrial premature beat",
        "a": "Aberrated atrial premature beat",
        "J": "Nodal (junctional) premature beat",
        "S": "Supraventricular premature beat",
        9: "Premature ventricular contraction",
        "F": "Fusion of ventricular and normal beat",
        "[": "Start of ventricular flutter/fibrillation",
        "!": "Ventricular flutter wave",
        "]": "End of ventricular flutter/fibrillation",
        "e": "Atrial escape beat",
        "j": "Nodal (junctional) escape beat",
        "E": "Ventricular escape beat",
        "/": "Paced beat",
        "f": "Fusion of paced and normal beat",
        "x": "Non-conducted P-wave (blocked APB)",
        "Q": "Unclassifiable",
        "|": "Isolated QRS-like artifact"
    }
    return annotation_mapping.get(pred_value, "Unknown")

def get_rhythm_label(prediction):
    """
    The function `get_rhythm_label` takes a prediction value and returns the corresponding rhythm label
    based on a mapping.
    
    :param prediction: A list containing the predicted value for the rhythm classification
    :return: the rhythm label based on the prediction value. If the prediction value is 0, it returns
    "Normal sinus rhythm". If the prediction value is 1, it returns "Sinus bradycardia". If the
    prediction value is 2, it returns "Ventricular tachycardia". If the prediction value is not 0, 1, or
    """
    pred_value = prediction[0]
    rhythm_annotation_mapping = {
        0: "Normal Sinus Rhythm",
        1: "Sinus Bradycardia", 
        2: "Ventricular Tachycardia"
    }
    return rhythm_annotation_mapping.get(pred_value, "Unknown")

def create_dataframe(features, csv_file_path):
    """
    The function creates a new row of data from a list of features and appends it to an existing CSV
    file or creates a new CSV file if it doesn't exist.
    
    :param features: The "features" parameter is a list or array containing the values for each feature
    or column in the dataframe. Each element in the list represents a value for a specific feature
    :param csv_file_path: The file path where the CSV file will be saved or updated
    """
    new_row = pd.DataFrame([features])
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(csv_file_path, index=False)
