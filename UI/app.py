#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 13:34:17 2023

@author: ivanlorenzanabelli
"""
import numpy as np
import wfdb
import pywt
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, lfilter, find_peaks
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QColor
from PyQt5.QtWidgets import QWidget, QMessageBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QGraphicsDropShadowEffect
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from draggableLabel import DraggableLabel
import ecg_feature_extractor as ecg_features

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initAttributes()
        self.initUI()
        
    def initAttributes(self):
        self.window_size = 1200
        self.ecgGraphed = False  
        self.current_position = 0
        self.ECGSeries = None
        self.heaFilePath = None
        self.datFilePath = None
        self.update_interval = 50
        self.scroll_amount = 10
        self.currentWindowIndex = -1 
        #self.initTimer()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        # Agregar sombra
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(3)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 80))
        
        # Crear un QHBoxLayout para la subida de archivos
        fileLayout = QHBoxLayout()
        
        # Botón de subida para archivo .hea
        self.fileLabelHEA = DraggableLabel("Drag and drop or click to upload .hea file")
        self.fileLabelHEA.setStyleSheet("border: 2px dashed white; padding: 20px;")
        self.fileLabelHEA.setAcceptDrops(True)
        self.fileLabelHEA.mousePressEvent = self.labelClickedHEA
        self.fileLabelHEA.setMinimumSize(525, 75)  # Ancho mínimo, Alto mínimo
        self.fileLabelHEA.setMaximumSize(525, 75)  # Ancho máximo, Alto máximo
        fileLayout.addWidget(self.fileLabelHEA)
        
        # Botón de subida para archivo .dat
        self.fileLabelDAT = DraggableLabel("Drag and drop or click to upload .dat file")
        self.fileLabelDAT.setStyleSheet("border: 2px dashed white; padding: 20px;")
        self.fileLabelDAT.setAcceptDrops(True)
        self.fileLabelDAT.mousePressEvent = self.labelClickedDAT
        self.fileLabelDAT.setMinimumSize(525, 75)  # Ancho mínimo, Alto mínimo
        self.fileLabelDAT.setMaximumSize(525, 75)  # Ancho máximo, Alto máximo
        fileLayout.addWidget(self.fileLabelDAT)
        
        # Agregar el QHBoxLayout al layout principal
        layout.addLayout(fileLayout)

        # Canvas para graficar la señal
        self.canvas = FigureCanvas(plt.Figure(figsize=(5, 3)))
        self.canvas.setGraphicsEffect(shadow)
        layout.addWidget(self.canvas,3)
        
        # Espacio para el diagnóstico
        self.diagnosisSpace = QWidget()
        self.diagnosisSpace.setStyleSheet("border: 1px solid gray; height: 150px;")
        self.diagnosisSpace.setStyleSheet("background-color: white;")
        
        # Layout para los botones
        buttonsLayout = QVBoxLayout()
        self.button1 = QPushButton("Previous")
        self.button1.setMinimumSize(200, 35)
        self.button1.setMaximumSize(200, 35)
        self.button1.clicked.connect(self.prevWindow)
        self.button1.setEnabled(False)  # Inicialmente deshabilitado
        self.button1.setGraphicsEffect(shadow)
        self.button1.setStyleSheet(
            "QPushButton:enabled { background-color: #FF6347; color: white; border-radius: 10px; }"
            "QPushButton:disabled { background-color: #A04030; color: white; border-radius: 10px; opacity: 0.8;}"
            "QPushButton:hover:enabled { background-color: #FF4500; }"
        )
        self.button2 = QPushButton("Next")
        self.button2.setMinimumSize(200, 35)
        self.button2.setMaximumSize(200, 35)
        self.button2.clicked.connect(self.nextWindow)
        self.button2.setEnabled(False)  # Inicialmente deshabilitado
        self.button2.setGraphicsEffect(shadow)
        self.button2.setStyleSheet(
            "QPushButton:enabled { background-color: #32CD32; color: white; border-radius: 10px; }"
            "QPushButton:disabled { background-color: #206820; color: white; border-radius: 10px; opacity: 0.8;}"
            "QPushButton:hover:enabled { background-color: #228B22; }"
        )
        self.button2.setGraphicsEffect(shadow)
        buttonsLayout.addWidget(self.button1)
        buttonsLayout.addWidget(self.button2)
        
        # Layout para las etiquetas
        labelsLayout = QVBoxLayout()
        self.resultsLabel = QLabel("Diagnosis Results")
        self.resultsLabel.setStyleSheet("border: none; font-weight: bold;")
        labelsLayout.addWidget(self.resultsLabel)
        
        self.beatResult = QLabel("Beat Detected: ")
        self.beatResult.setStyleSheet("border: none;")
        labelsLayout.addWidget(self.beatResult)
        
        self.rhythmResult = QLabel("Rythm Diagnosed: ")
        self.rhythmResult.setStyleSheet("border: none;")
        labelsLayout.addWidget(self.rhythmResult)
        
        # Layout horizontal que contiene los dos QVBoxLayouts
        diagnosisLayout = QHBoxLayout(self.diagnosisSpace)
        diagnosisLayout.addLayout(labelsLayout)
        diagnosisLayout.addLayout(buttonsLayout)
        
        # Agregar el layout horizontal al widget principal
        layout.addWidget(self.diagnosisSpace)

        
        # Crear un QHBoxLayout para los botones
        buttonLayout = QHBoxLayout()
    
        # Botón para graficar
        self.plotButton = QPushButton("Plot ECG")
        self.plotButton.clicked.connect(self.plotECG)
        self.plotButton.setEnabled(False)  # Inicialmente deshabilitado
        self.plotButton.setGraphicsEffect(shadow)
        buttonLayout.addWidget(self.plotButton)
        
        # Botón para diagnosticar
        self.diagnoseButton = QPushButton("Diagnose")
        self.diagnoseButton.clicked.connect(self.startDiagnosis)
        self.diagnoseButton.setEnabled(False)  # Inicialmente deshabilitado
        self.diagnoseButton.setGraphicsEffect(shadow)
        buttonLayout.addWidget(self.diagnoseButton)
        
        # Botón para borrar archivos
        self.deleteFileButtton = QPushButton("Delete Files")
        self.deleteFileButtton.clicked.connect(self.deleteFiles)
        self.deleteFileButtton.setEnabled(False)  # Inicialmente deshabilitado
        self.deleteFileButtton.setGraphicsEffect(shadow)
        buttonLayout.addWidget(self.deleteFileButtton)
        
        
        # Agregar el QHBoxLayout al layout principal
        layout.addLayout(buttonLayout)
        pixmap = QPixmap('/Users/ivanlorenzanabelli/Arrhythmia-Detector/UI/cool-background3.png')
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)

        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)
        self.setFixedSize(1100, 600)
        self.setLayout(layout)
        self.setWindowTitle('ECG Arrhythmia Analyzer')
        self.show()

    # Funciones para manejo de archivos y eventos de usuario
    def uploadFile(self, fileType):
        options = QFileDialog.Options()
        if fileType == "hea":
            fileLabel = self.fileLabelHEA
            fileExtension = "HEA Files (*.hea)"
        elif fileType == "dat":
            fileLabel = self.fileLabelDAT
            fileExtension = "DAT Files (*.dat)"
        else:
            return  # O manejar el error
    
        fileName, _ = QFileDialog.getOpenFileName(self, f"Seleccione archivo {fileType.upper()}", "", fileExtension, options=options)
        if fileName:
            fileLabel.setText(fileName)
            if fileType == "hea":
                self.heaFilePath = fileName
            elif fileType == "dat":
                self.datFilePath = fileName
            self.checkFilesAndEnablePlotButton()
        
    def checkFilesAndEnablePlotButton(self):
        # Habilitar el botón de Graficar solo si ambos archivos están presentes
        if self.heaFilePath is not None and self.datFilePath is not None:
            self.plotButton.setEnabled(True)
        else:
            self.plotButton.setEnabled(False)
        
        # Habilitar el botón de Diagnóstico solo si la señal ECG ha sido graficada
        if self.ecgGraphed:
            self.diagnoseButton.setEnabled(True)
        else:
            self.diagnoseButton.setEnabled(False)
    
        # Habilitar el botón de Borrar Archivos si al menos uno de los archivos está presente
        if self.heaFilePath is not None or self.datFilePath is not None:
            self.deleteFileButtton.setEnabled(True)
        else:
            self.deleteFileButtton.setEnabled(False)
    
    def handleFile(self, file_path):
        if file_path.endswith(".hea"):
            self.fileLabelHEA.setText(file_path)
            self.heaFilePath = file_path
        elif file_path.endswith(".dat"):
            self.fileLabelDAT.setText(file_path)
            self.datFilePath = file_path
        self.checkFilesAndProcess()
            
    # Funciones para procesamiento y visualización de datos
    def loadAndProcessFiles(self):
        # Implementa aquí la lógica para leer y procesar los archivos .hea y .dat
        try:
            self.processECGFiles(self.heaFilePath, self.datFilePath)
            #self.ECGSeries = ... # Carga la serie ECG desde los archivos
            #self.plot(self.ECGSeries[self.current_position:self.current_position+254], self.current_position)
        except Exception as e:
            self.showErrorAlert(str(e))
         
    def plotECG(self):
        try:
            # Cargar la señal ECG
            record = wfdb.rdrecord(self.heaFilePath.replace('.hea', ''))
            signal = record.p_signal[:, 0]

            # Procesamiento básico de la señal
            signal_centered = signal - np.mean(signal)
            b, a = iirnotch(60, 30, record.fs)
            signal_filtered = lfilter(b, a, signal_centered)

            # Detectar los picos R
            peaks, _ = find_peaks(signal_filtered, distance=int(0.7 * record.fs))

            # Almacenar las ventanas y los picos
            self.windows = [ecg_features.apply_window(signal_filtered, peak, record.fs) for peak in peaks]
            self.peaks = peaks
            self.signal_filtered = signal_filtered

            # Graficar toda la señal con el límite de eje X especificado
            self.plotFullECG()

            # Actualizar estado y botones
            self.checkFilesAndEnablePlotButton()
            self.plotButton.setEnabled(False)
            self.fileLabelHEA.setEnabled(False)
            self.fileLabelDAT.setEnabled(False)
            self.button2.setEnabled(True)
        except Exception as e:
            self.showErrorAlert(str(e))

    def plotFullECG(self):
        # Limpiar el canvas actual y obtener el contexto de dibujo
        self.canvas.figure.clear()
        ax = self.canvas.figure.subplots()

        # Dibujar en el canvas la señal completa con límite de eje X
        ax.plot(self.signal_filtered[:1200])

        # Configurar los detalles de la gráfica
        ax.set_title('Full ECG Signal')
        ax.set_xlabel('Samples (360 samples = 1 second)')
        ax.set_ylabel('Amplitude')

        # Actualizar el canvas
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plotWindow(self, windowIndex):
        # Obtener la ventana actual
        window = self.windows[windowIndex]

        # Normalizar la ventana actual
        window_normalized = ecg_features.min_max_normalize(window)

        # Limpiar el canvas actual y obtener el contexto de dibujo
        self.canvas.figure.clear()
        ax = self.canvas.figure.subplots()

        # Dibujar en el canvas
        ax.plot(window_normalized)

        # Configurar los detalles de la gráfica
        ax.set_title(f'ECG Window {windowIndex + 1}/{len(self.windows)}')
        ax.set_xlabel('Samples (360 samples = 1 second)')
        ax.set_ylabel('Amplitude')

        # Actualizar el canvas
        self.canvas.figure.tight_layout()
        self.canvas.draw()


    # Funciones para la gestión de la interfaz gráfica
    def plot(self, data, start_position, peaks=None):
        self.canvas.figure.clear()
        ax = self.canvas.figure.subplots()
    
        # Asegurarse de que no se exceda el final de la serie de datos
        end_position = min(start_position + self.window_size, len(data))
        segment = data[start_position:end_position]
    
        ax.plot(range(start_position, end_position), segment)
    
        if peaks is not None:
            # Ajustar picos para el segmento actual
            peaks_in_segment = [p for p in peaks if start_position <= p < end_position]
            ax.plot([p for p in peaks_in_segment], [segment[p - start_position] for p in peaks_in_segment], "x")
    
        ax.set_title('ECG Signal')
        ax.set_xlabel('Samples (360 samples = 1 second)')
        ax.set_ylabel('Amplitude')
        ax.set_ylim(0, 1.2) 
        ax.set_ylim(0, 1200)
        self.canvas.figure.tight_layout()  # Ajustar el layout del gráfico
        self.canvas.draw()

    def nextWindow(self):
        # Incrementar el índice de ventana y mostrar la siguiente ventana, solo si hay más ventanas disponibles
        if self.currentWindowIndex + 1 < len(self.windows):
            self.currentWindowIndex += 1
            self.plotWindow(self.currentWindowIndex)
            
            # Actualizar etiquetas de diagnóstico
            self.beatResult.setText("Beat Detected:")
            self.rhythmResult.setText("Rythm Diagnosed:")

            # Habilitar botón "Anterior" si no estamos en la primera ventana
            self.button1.setEnabled(self.currentWindowIndex > 0)

        # Habilitar botón "Diagnóstico" si la señal ECG ha sido graficada
        if not self.ecgGraphed:
            self.ecgGraphed = True
            self.diagnoseButton.setEnabled(True)

    def prevWindow(self):
        # Decrementar el índice de ventana y mostrar la ventana anterior
        if self.currentWindowIndex - 1 >= 0:
            self.currentWindowIndex -= 1
            self.plotWindow(self.currentWindowIndex)
            self.beatResult.setText("Beat Detected:")
            self.rhythmResult.setText("Rythm Diagnosed:")

    # Funciones para mostrar alertas y mensajes
    def showErrorAlert(self, errorMessage):
        alert = QMessageBox()
        alert.setWindowTitle("Error de Procesamiento")
        alert.setText("Se produjo un error al procesar los archivos:\n" + errorMessage)
        alert.setIcon(QMessageBox.Critical)
        alert.exec_()
            
    def showMissingFileAlert(self):
        alert = QMessageBox()
        alert.setWindowTitle("Archivo Faltante")
        alert.setText("Por favor, asegúrese de haber seleccionado ambos archivos: .hea y .dat")
        alert.setIcon(QMessageBox.Warning)
        alert.exec_()
        
    # Event Handling
    def labelClickedHEA(self, event):
        self.uploadFile("hea")

    def labelClickedDAT(self, event):
        self.uploadFile("dat")
        
    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def deleteFiles(self):
        # Eliminar las referencias a los archivos
        self.heaFilePath = None
        self.datFilePath = None

        # Actualizar las etiquetas de los archivos a su estado original
        self.fileLabelHEA.setText("Drag and drop or click to upload .hea file")
        self.fileLabelDAT.setText("Drag and drop or click to upload .dat file")
        
        # Volver a habilitar los botones de subida de archivos
        self.fileLabelHEA.setEnabled(True)
        self.fileLabelDAT.setEnabled(True)

        # Deshabilitar el botón Graficar, Diagnosticar y Borrar
        self.ecgGraphed = False
        self.plotButton.setEnabled(False)
        self.diagnoseButton.setEnabled(False)
        self.deleteFileButtton.setEnabled(False)
        self.button1.setEnabled(False)
        self.button2.setEnabled(False)

        # Opcional: Limpiar el gráfico actual, si es necesario
        self.canvas.figure.clear()
        self.canvas.draw()

        # Opcional: Detener cualquier proceso en ejecución relacionado con los archivos
        self.current_position = 0
        self.currentWindowIndex = 0 
        self.ECGSeries = None

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if file_path.endswith('.hea'):
            self.handleHEAFile(file_path)
        elif file_path.endswith('.dat'):
            self.handleDATFile(file_path)
        
    # Control de Flujo de la Aplicación
    def checkFilesAndProcess(self):
        if hasattr(self, 'heaFilePath') and hasattr(self, 'datFilePath'):
            self.loadAndProcessFiles()
        else:
            self.showMissingFileAlert()

    def startDiagnosis(self):
        if self.currentWindowIndex < len(self.windows):
            current_window = self.windows[self.currentWindowIndex]

            features = ecg_features.extract_features_from_window(current_window, 360)
            predictionB, predictionR = ecg_features.predict_ecg(features)

            self.updateDiagnosisUI(predictionB,predictionR)

    def updateDiagnosisUI(self, predictionB, predictionR):
        self.beatResult.setText(f"Beat Detected: {predictionB}")
        self.rhythmResult.setText(f"Rythm Diagnosed: {predictionR}")
