#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 13:34:17 2023

@author: ivanlorenzanabelli
"""

import wfdb
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QColor
from PyQt5.QtWidgets import QWidget, QMessageBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QGraphicsDropShadowEffect
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from draggableLabel import DraggableLabel
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, lfilter, find_peaks

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
        self.initTimer()
        
    def initTimer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advanceSignal)
    
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
        self.fileLabelHEA = DraggableLabel("Arrastra y suelta o haz click para subir archivo .hea")
        self.fileLabelHEA.setStyleSheet("border: 2px dashed white; padding: 20px;")
        self.fileLabelHEA.setAcceptDrops(True)
        self.fileLabelHEA.mousePressEvent = self.labelClickedHEA
        self.fileLabelHEA.setMinimumSize(525, 75)  # Ancho mínimo, Alto mínimo
        self.fileLabelHEA.setMaximumSize(525, 75)  # Ancho máximo, Alto máximo
        fileLayout.addWidget(self.fileLabelHEA)
        
        # Botón de subida para archivo .dat
        self.fileLabelDAT = DraggableLabel("Arrastra y suelta o haz click para subir archivo .dat")
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
        
        diagnosis_layout = QVBoxLayout(self.diagnosisSpace)
        
        # Resultados del modelo
        self.resultsLabel = QLabel("Resultados del Modelo: ")
        self.resultsLabel.setStyleSheet("border: none;")
        diagnosis_layout.addWidget(self.resultsLabel)
        
        self.diagnosisResult = QLabel("Diagnostico: ")
        self.diagnosisResult.setStyleSheet("border: none;")
        diagnosis_layout.addWidget(self.diagnosisResult)
        
        layout.addWidget(self.diagnosisSpace)
        
        # Crear un QHBoxLayout para los botones
        buttonLayout = QHBoxLayout()
    
        # Botón para graficar
        self.plotButton = QPushButton("Graficar")
        self.plotButton.clicked.connect(self.plotECG)
        self.plotButton.setEnabled(False)  # Inicialmente deshabilitado
        self.plotButton.setGraphicsEffect(shadow)
        buttonLayout.addWidget(self.plotButton)
        
        # Botón para diagnosticar
        self.diagnoseButton = QPushButton("Diagnosticar")
        self.diagnoseButton.clicked.connect(self.startDiagnosis)
        self.diagnoseButton.setEnabled(False)  # Inicialmente deshabilitado
        self.diagnoseButton.setGraphicsEffect(shadow)
        buttonLayout.addWidget(self.diagnoseButton)
        
        # Botón para borrar archivos
        self.deleteFileButtton = QPushButton("Borrar Archivos")
        self.deleteFileButtton.clicked.connect(self.deleteFiles)
        self.deleteFileButtton.setEnabled(False)  # Inicialmente deshabilitado
        self.deleteFileButtton.setGraphicsEffect(shadow)
        buttonLayout.addWidget(self.deleteFileButtton)
        
        
        # Agregar el QHBoxLayout al layout principal
        layout.addLayout(buttonLayout)
        
        pixmap = QPixmap('cool-background3.png')
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)

        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)
        self.setFixedSize(1100, 600)
        self.setLayout(layout)
        self.setWindowTitle('Diagnóstico de Arritmia')
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
            print(f"Error: {e}")
            self.showErrorAlert(str(e))
         
    def plotECG(self):
        try:
            # Cargar la señal ECG
            record = wfdb.rdrecord(self.heaFilePath.replace('.hea', ''))
            signal = record.p_signal[:, 0]
    
            # Procesamiento básico de la señal
            signal_centered = signal - np.mean(signal)
            signal_normalized = signal_centered / np.max(np.abs(signal_centered))
            b, a = iirnotch(60, 30, record.fs)
            signal_filtered = lfilter(b, a, signal_normalized)
    
            # Detectar los picos R
            peaks, _ = find_peaks(signal_filtered, distance=int(0.7 * record.fs))
    
            # Almacenar la señal filtrada para visualización
            self.ECGSeries = signal_filtered
    
            # Graficar la señal ECG con picos R
            self.plot(self.ECGSeries, 0, peaks)
    
            # Actualizar estado y botones
            self.ecgGraphed = True
            self.checkFilesAndEnablePlotButton()
            self.plotButton.setEnabled(False)
            self.fileLabelHEA.setEnabled(False)
            self.fileLabelDAT.setEnabled(False)
    
        except Exception as e:
            print(f"Error al procesar los archivos: {e}")
            self.showErrorAlert(str(e))

            
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
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')
        self.canvas.draw()

    def advanceSignal(self):
        if self.ECGSeries is not None:
            self.current_position += self.scroll_amount  # O cualquier otro valor que desees para el avance

            # Verificar que no se haya llegado al final de la señal
            if self.current_position + self.window_size >= len(self.ECGSeries):
                self.timer.stop()
                return

            self.plot(self.ECGSeries, self.current_position)

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
        self.fileLabelHEA.setText("Arrastra y suelta o haz click para subir archivo .hea")
        self.fileLabelDAT.setText("Arrastra y suelta o haz click para subir archivo .dat")
        
        # Volver a habilitar los botones de subida de archivos
        self.fileLabelHEA.setEnabled(True)
        self.fileLabelDAT.setEnabled(True)

        # Deshabilitar el botón Graficar, Diagnosticar y Borrar
        self.ecgGraphed = False
        self.plotButton.setEnabled(False)
        self.diagnoseButton.setEnabled(False)
        self.deleteFileButtton.setEnabled(False)

        # Opcional: Limpiar el gráfico actual, si es necesario
        self.canvas.figure.clear()
        self.canvas.draw()

        # Opcional: Detener cualquier proceso en ejecución relacionado con los archivos
        self.timer.stop()
        self.current_position = 0
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
        if self.ECGSeries is not None:
            self.timer.start(self.update_interval)