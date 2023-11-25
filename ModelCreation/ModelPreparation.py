#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:20:58 2023

@author: ivanlorenzanabelli
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from sineWave import boxcar
from scipy.signal import iirnotch, lfilter, find_peaks
import pywt

# Windows
DATASET_DIRECTORY = 'C:\\Users\\XPG\\Desktop\\Sotelo\\ProyectoFinal\\mit-bih-arrhythmia-database-1.0.0\\Dataset\\Entrenamiento'
#MODEL_SAVE_DIRECTORY = 'C:\\Users\\XPG\\Desktop\\Sotelo\\ProyectoFinal\\Source Code'
CSV_FILE_PATH = 'C:\\Users\\XPG\\Desktop\\Sotelo\\ProyectoFinal\\Source Code\\ecg_features3.csv'
# Mac
#DATASET_DIRECTORY = '/Users/ivanlorenzanabelli/Desktop/Diagnostico Asistido/Proyecto Final/Dataset50'
#CSV_FILE_PATH = '/Users/ivanlorenzanabelli/Desktop/Diagnostico Asistido/ProyectoPrueba/ecg_features1.csv'

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
    fft_freq = np.fft.rfftfreq(len(signal_windowed), 1/fs)
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

def create_dataframe(features):
    """
    The function `create_dataframe` reads a CSV file, adds a new row with the given features, and saves
    the updated DataFrame back to the CSV file.
    
    :param features: The "features" parameter is a list or array containing the values for each feature
    that you want to add as a new row to the DataFrame. Each element in the list corresponds to a
    feature in the DataFrame
    """
    df = pd.read_csv(CSV_FILE_PATH)
    # Create a new DataFrame from your features
    new_row = pd.DataFrame([features])
    # Use concat to add the new row
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE_PATH, index=False)

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

def main():
    # Carga los nombres de archivos del dataset
    hea_files = [f for f in os.listdir(DATASET_DIRECTORY) if f.endswith('.hea')]
    record_names = [os.path.splitext(f)[0] for f in hea_files]
    record_names.sort()
    
    # annotation_mapping = {
    #     "N": "Normal",
    #     "·": "Normal",
    #     "L": "Left bundle branch block beat",
    #     "R": "Right bundle branch block beat",
    #     "A": "Atrial premature beat",
    #     "a": "Aberrated atrial premature beat",
    #     "J": "Nodal (junctional) premature beat",
    #     "S": "Supraventricular premature beat",
    #     "V": "Premature ventricular contraction",
    #     "F": "Fusion of ventricular and normal beat",
    #     "[": "Start of ventricular flutter/fibrillation",
    #     "!": "Ventricular flutter wave",
    #     "]": "End of ventricular flutter/fibrillation",
    #     "e": "Atrial escape beat",
    #     "j": "Nodal (junctional) escape beat",
    #     "E": "Ventricular escape beat",
    #     "/": "Paced beat",
    #     "f": "Fusion of paced and normal beat",
    #     "x": "Non-conducted P-wave (blocked APB)",
    #     "Q": "Unclassifiable",
    #     "|": "Isolated QRS-like artifact"
    # }
    
    annotation_mapping = {
        # Beats Normales y Bloqueos de Rama
        "N": 0, "L": 1, "R": 2, "e": 3, "j": 4,
        # Beats Atriales
        "A": 5, "a": 6, "S": 7,
        # Beats Nodales
        "J": 8, 
        # Beats Ventriculares
        "V": 9, "E": 10, "F": 11, "!": 12,
        # Beats Marcados (Paced)
        #"/": 13, "f": 14
        # Beats No Clasificables y Artefactos
        #"x": 15, "Q": 16, "|": 17
    }


    # "(N": "Normal sinus rhythm", 0
    # "(SBR": "Sinus bradycardia", 1
    # "(VT": "Ventricular tachycardia", 3
    
    rhythm_annotation_mapping = {
        "(N": 0, "(SBR": 1, "(VT": 2
    }
    
    # Procesar cada registro
    for record_name in record_names:
        print(record_name)
        record_path = os.path.join(DATASET_DIRECTORY, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        rhythm_annotations = wfdb.rdann(record_path, 'atr')
        rhythm_labels = []
    
        with open(record_path + '.hea', 'r') as header_file:
            header_lines = header_file.readlines()
    
        ml_ii_index = get_ml_ii_index(header_lines)
        current_rhythm_label = 'Unknown'
        
        # Iterate over all annotations
        for ann_index, ann_sample in enumerate(rhythm_annotations.sample):
            ann_label = rhythm_annotations.aux_note[ann_index].rstrip('\x00')
            if ann_label == '':
                pass
            elif ann_label in rhythm_annotation_mapping:
                current_rhythm_label = rhythm_annotation_mapping[ann_label]
            else:
                current_rhythm_label = "Unknown"
            rhythm_labels.append(current_rhythm_label)

        if ml_ii_index is not None:
            signal = record.p_signal[:, ml_ii_index]
        else:
            print(f"MLII channel not found in {record_name}.")
            continue
    
        signal_centered = signal - np.mean(signal)
        fs = record.fs
        b, a = iirnotch(60, 30, fs)
        signal_filtered = lfilter(b, a, signal_centered)
    
        # Detect R peaks in ECG signal
        peaks, _ = find_peaks(signal_filtered, distance=int(0.7 * fs))
        peak_times = peaks / fs
    
        # Find closest annotations to R peaks
        closest_annotations = [annotation_mapping.get(annotation.symbol[np.argmin(np.abs(annotation.sample / fs - pt))], "Unknown") for pt in peak_times]
        closest_rhythm_labels = [rhythm_labels[np.argmin(np.abs(rhythm_annotations.sample / fs - pt))] for pt in peak_times]
    
        width = 1  # Window width in seconds
        print(len(peaks))
        for i, peak in enumerate(peaks):
            print(i)
            signal_windowed = apply_window(signal_filtered, peak, width, fs)
    
            if signal_windowed is not None:
                signal_windowed_normalized = min_max_normalize(signal_windowed)
                distanceW = int(0.45 * fs)
                peaksW, _ = find_peaks(signal_windowed_normalized, distance=distanceW)
                peaksW = peaksW[signal_windowed_normalized[peaksW] > 0.6]
                std_val = np.std(signal_windowed_normalized)
                
                rhythm_label = closest_rhythm_labels[i]
                annotation_label = closest_annotations[i] if i < len(closest_annotations) else "Unknown"
                
                #plt.figure(figsize=(8, 3))
                #plt.plot(signal_filtered, label = 'ECG') # Selecciona el canal MLII (columna 0)
                #plt.plot(peaks, signal_filtered[peaks], 'rx', label = 'Picos R')
                # plt.plot(signal_windowed_normalized, '--') # Selecciona el canal MLII (columna 0)
                # plt.plot(peaksW, signal_windowed_normalized[peaksW], 'rx', label = 'Picos R')
                # # for time in annotation.sample:
                # #     plt.axvline(x=time, color='r', linestyle='--')
                # plt.title('ECG Signal')
                # plt.xlabel('Muestras')
                # plt.ylabel('Amplitud')
                # plt.axis([0, 1000, -0.2, 1.1]) #make visible slower frequency period
                # plt.show()
                
                # FFT and Wavelet Transform
                total_spectral_energy, total_psd, dominant_freq, wavelet_Energy, total_shannon_entropy = calculate_fft_and_wavelet(signal_windowed_normalized, record.fs)
    
                # Extract features for the current signal in the window
                features = {
                    "RPeakCount": len(peaksW),
                    "MaxRAmplitude": np.max(signal_windowed_normalized) if len(signal_windowed_normalized) > 0 else 0,
                    "SpectralEnergy": total_spectral_energy,
                    "TotalPSD": total_psd,
                    "WaveletEnergy": wavelet_Energy,
                    "ShannonEntropy": total_shannon_entropy,
                    "SignalSTD": std_val,
                    "BeatType": annotation_label,
                    "RhythmClass": rhythm_label
                }
    
                create_dataframe(features)

# Entry point
if __name__ == "__main__":
    main()