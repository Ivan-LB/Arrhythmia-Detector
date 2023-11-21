#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:20:58 2023

@author: ivanlorenzanabelli
"""

import os
import numpy as np
import pandas as pd
import wfdb
from sineWave import boxcar
from scipy.signal import iirnotch, lfilter, find_peaks
import pywt

DATASET_DIRECTORY = 'C:\\Users\\XPG\\Desktop\\Sotelo\\ProyectoFinal\\mit-bih-arrhythmia-database-1.0.0\\Dataset\\No usar'
MODEL_SAVE_DIRECTORY = 'C:\\Users\\XPG\\Desktop\\Sotelo\\ProyectoFinal\\Source Code'
CSV_FILE_PATH = 'C:\\Users\\XPG\\Desktop\\Sotelo\\ProyectoFinal\\Source Code\\ecg_features2.csv'

# Function definitions
def get_demographics(header_lines):
    """
    The function `get_demographics` extracts age and gender information from a list of header lines and
    returns them as a tuple.
    
    :param header_lines: A list of strings representing the header lines of a data file. Each string in
    the list represents a line of the header
    :return: the age and gender values extracted from the header lines.
    """
    for line in header_lines:
        if line.startswith('#'):
            parts = line.split()
            # Assuming age and gender are in positions 2 and 3 respectively
            age = parts[1] if len(parts) > 3 else None  # El índice debe ser 1 para la edad
            gender_str = parts[2] if len(parts) > 4 else None  # El índice debe ser 2 para el género
            # Convertir el género a un valor numérico
            gender = int(0) if gender_str == 'M' else int(1) if gender_str == 'F' else None
            return age, gender
    return None, None  # In case the information is not found

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
    # Calculate the FFT and FFT power
    fft_coeffs = np.fft.rfft(signal_windowed)
    fft_freq = np.fft.rfftfreq(len(signal_windowed), 1/fs)
    fft_power = np.abs(fft_coeffs)**2
    
    # Power and Dominant Frequency
    total_power = np.sum(fft_power)
    dominant_freq = fft_freq[np.argmax(fft_power)]
    
    # Parameters for wavelets - Adjusted for a broader range of scales
    wavelet_name = 'mexh'
    maxScale = 16  # Example max scale, adjust based on your signal's characteristics
    minScale = 1
    scaleStep = 1  # Example scale step, adjust as needed

    # Wavelet Transform with adjusted scales
    scales = np.arange(minScale, maxScale + scaleStep, scaleStep)
    coefc, freqs = pywt.cwt(signal_windowed, scales, wavelet_name)
    wavelet_Energy = np.sum(coefc**2)

    # Energy of the wavelet coefficients
    energy = np.sum(coefc**2, axis=1)
    # Ensure no division by zero
    energy[energy == 0] = np.finfo(float).eps

    # Convert coefficients to a probability distribution
    prob_dist = coefc**2 / energy[:, None]
    # Normalize to ensure it sums to 1
    prob_dist /= np.sum(prob_dist, axis=1)[:, None]

    # Calculate Shannon entropy for each scale
    shannon_entropy = -np.sum(prob_dist * np.log2(prob_dist), axis=1)
    # Calculate total entropy
    total_shannon_entropy = np.mean(shannon_entropy)
    return total_power, dominant_freq, wavelet_Energy, total_shannon_entropy

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
    # Tiempo inicial del segmento de interés basado en el pico R actual
    start_index = max(peak - int(width/2 * fs), 0)
    end_index = min(peak + int(width/2 * fs), len(signal))

    # Asegurarse de que no estemos intentando crear una ventana con una longitud negativa
    if start_index < end_index:
        # Extraer la ventana de la señal
        signal_windowed = signal[start_index:end_index]
        return signal_windowed
    return None

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
        # Ritmos Normales y Bloqueos de Rama
        "N": 0,
        "L": 0,
        "R": 0,
        "e": 0,
        "j": 0,
        
        # Ritmos Atriales
        "A": 1,
        "a": 1,
        "S": 1,
        
        # Ritmos Nodales
        "J": 2,
        
        # Ritmos Ventriculares
        "V": 3,
        "E": 3,
        "F": 3,
        "!": 3,
        
        # Ritmos Marcados (Paced)
        "/": 4,
        "f": 4,
        
        # Ritmos No Clasificables y Artefactos
        "x": 5,
        "Q": 6,
        "|": 5
    }

    # "(N": "Normal sinus rhythm", 0
    # "(P": "Paced rhythm", 1
    # "(SBR": "Sinus bradycardia", 2
    # "(SVTA": "Supraventricular tachyarrhythmia", 3
    # "(VT": "Ventricular tachycardia" 4
    
    rhythm_annotation_mapping = {
        "(N": 0,
        "(P": 1,
        "(SBR": 2,
        "(SVTA": 3,
        "(VT": 4
    }
    
    # Procesar cada registro
    for record_name in record_names:
        print(record_name)
        record_path = os.path.join(DATASET_DIRECTORY, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        # Leer las anotaciones de ritmo
        rhythm_annotations = wfdb.rdann(record_path, 'atr')
        rhythm_labels = []
        
        with open(record_path + '.hea', 'r') as header_file:
            header_lines = header_file.readlines()
        
        ml_ii_index = get_ml_ii_index(header_lines)
        age, gender = get_demographics(header_lines)
        
        current_rhythm_label = 'Unknown'
        
        # Iterar sobre todas las anotaciones
        for ann_index, ann_sample in enumerate(rhythm_annotations.sample):
            ann_label = rhythm_annotations.aux_note[ann_index].rstrip('\x00')  # Eliminar '\x00'
            # Comprueba si la anotación es una cadena vacía y, si lo es, simplemente continúa con el último valor conocido
            if ann_label == '':
                pass
            # Si la anotación actual es una anotación de ritmo y está en el mapeo, actualiza current_rhythm_label
            elif ann_label in rhythm_annotation_mapping:
                current_rhythm_label = rhythm_annotation_mapping[ann_label]
            # Si la anotación no está en el mapeo y no es una cadena vacía, establece current_rhythm_label a "Unknown"
            else:
                current_rhythm_label = "Unknown"
            # Añade la etiqueta de ritmo actual a la lista rhythm_labels
            rhythm_labels.append(current_rhythm_label)

        # Verificar si se encontró el índice del canal MLII
        if ml_ii_index is not None:
            # Extraer la señal del canal MLII
            signal = record.p_signal[:, ml_ii_index]
        else:
            # Manejar el caso en el que no se encuentre el canal MLII
            print(f"El canal MLII no se encontró en el registro {record_name}.")
            continue  # Puedes decidir continuar con el siguiente registro o tomar otra acción
        # Procesamiento de la señal
        
        signal_centered = signal - np.mean(signal)
        signal_normalized = signal_centered / np.max(np.abs(signal_centered))
        
        # Aplicar Fitro Notch para remover las frecuencias de 50/60Hz que puedan interferir
        fs = record.fs
        b, a = iirnotch(60, 30, fs)
        signal_filtered = lfilter(b, a, signal_normalized)
        
        # Detectar los picos R de la señal de ECG
        peaks, _ = find_peaks(signal_filtered, distance=int(0.7 * fs))
        
        # Convertir los picos a indices de tiempo
        peak_times = peaks / fs

        # Encontrar la anotación más cercana a los picos R
        closest_annotations = []
        for peak_time in peak_times:
            closest_index = np.argmin(np.abs(annotation.sample / fs - peak_time))
            original_symbol = annotation.symbol[closest_index]
            # Map the original symbol to the normalized label
            normalized_label = annotation_mapping.get(original_symbol, "Unknown")
            closest_annotations.append(normalized_label)
            
        closest_rhythm_labels = []
        for peak_time in peak_times:
            closest_index = np.argmin(np.abs(rhythm_annotations.sample / fs - peak_time))
            closest_rhythm_label = rhythm_labels[closest_index]
            closest_rhythm_labels.append(closest_rhythm_label)

        # Ventana de análisis alrededor de cada pico R
        width = 1  # Ancho de la ventana en segundos
        print(len(peaks))
        for i, peak in enumerate(peaks):
            print(i)
            
            # Aplicar la ventana alrededor del pico R
            signal_windowed = apply_window(signal_filtered, peak, width, fs)
            
            if signal_windowed is not None:
                # Procesamiento de la señal en la ventana
                distanceW = int(0.45 * fs)  # Estimación del intervalo entre picos R (0.6 segundos)
                peaksW, _ = find_peaks(signal_windowed, distance=distanceW)
                peaksW = peaksW[signal_windowed[peaksW] > 0.2]
        
                std_val = np.std(signal_windowed)  # Desviación estándar
        
                # Encuentra la etiqueta de ritmo más cercana al pico R actual
                rhythm_label = closest_rhythm_labels[i]
                annotation_label = closest_annotations[i] if i < len(closest_annotations) else "Unknown"
        
                # Calcular la FFT y la Transformada Wavelet
                fft_power, dominant_freq, wavelet_Energy, total_shannon_entropy = calculate_fft_and_wavelet(signal_windowed, record.fs)
        
                # Extraer características para la señal actual en la ventana
                features = {
                    "picosR": len(peaksW),
                    "amplitudR": np.max(signal_windowed) if len(signal_windowed) > 0 else 0,
                    "potencia": fft_power,
                    "frecDominante": dominant_freq,
                    "energiaWavelet": wavelet_Energy,
                    "entropiaShannonW": total_shannon_entropy,
                    "sexo": gender,
                    "edad": age,
                    "std": std_val,
                    "tipoLatido": annotation_label,
                    "clase": rhythm_label
                }
        
                # Crear y actualizar DataFrame
                create_dataframe(features)

# Entry point
if __name__ == "__main__":
    main()