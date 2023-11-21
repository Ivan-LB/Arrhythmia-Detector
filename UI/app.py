#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 13:34:17 2023

@author: ivanlorenzanabelli
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QColor
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from draggableLabel import DraggableLabel

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.current_position = 0
        self.ECGSeries = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advanceSignal)
        self.update_interval = 100
        self.scroll_amount = 10
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        # Agregar sombra
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(3)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 80))
        
        # Botón de subida de archivo
        self.fileLabel = DraggableLabel("Arrastra y suelta o haz click para subir un archivo.")
        self.fileLabel.setStyleSheet("border: 2px dashed white; padding: 20px;")
        self.fileLabel.setAcceptDrops(True)
        self.fileLabel.mousePressEvent = self.labelClicked
        layout.addWidget(self.fileLabel)

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
        
        # Botón para diagnosticar
        self.diagnoseButton = QPushButton("Diagnosticar")
        self.diagnoseButton.clicked.connect(self.startDiagnosis)
        self.diagnosisSpace.setGraphicsEffect(shadow)
        layout.addWidget(self.diagnoseButton)
        
        pixmap = QPixmap('cool-background2.png')
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)

        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)
        self.setFixedSize(1100, 600)
        self.setLayout(layout)
        self.setWindowTitle('Diagnóstico de Arritmia')
        self.show()

    def uploadFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;TextFiles (*.txt);;Python Files (*.py)", options=options)
        if fileName:
            self.fileLabel.setText(fileName)
            self.timer.stop()
            self.current_position = 0
        try:
           self.ECGSeries = np.loadtxt(fileName, delimiter='\t')
           if self.ECGSeries.ndim > 1:
               self.ECGSeries = self.ECGSeries[:, 0]
           self.plot(self.ECGSeries[self.current_position:self.current_position+254],self.current_position)
        except Exception as e:
            print(f"Error al cargar o procesar el archivo: {e}")
            
    def labelClicked(self, event):
        self.uploadFile() 
        
    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        # Aquí se maneja la lógica cuando se suelta un archivo sobre el widget
        file_path = event.mimeData().urls()[0].toLocalFile()  # Obtiene la ruta del archivo
        self.handleFile(file_path)
        

    def plot(self, data, start_position):
        self.canvas.figure.clear()
        ax = self.canvas.figure.subplots()
        ax.plot(range(start_position, start_position + len(data)), data)  # Ajustar el rango x
        ax.set_title('ECG Signal')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        
        ax.set_xlim(start_position, start_position + len(data))  
        ax.set_ylim(-1.0, 2.0)
        self.canvas.draw()

    def advanceSignal(self):
        if self.ECGSeries is not None:
            self.current_position += self.scroll_amount 

        if self.current_position >= len(self.ECGSeries):
            self.timer.stop()
            return

        end_position = min(self.current_position + 254, len(self.ECGSeries))
        self.plot(self.ECGSeries[self.current_position:end_position],self.current_position)

    def startDiagnosis(self):
        if self.ECGSeries is not None:
            self.timer.start(self.update_interval)
            
    def handleFile(self, file_path):
        self.fileLabel.setText(file_path)
        self.timer.stop()
        self.current_position = 0
        try:
           self.ECGSeries = np.loadtxt(file_path, delimiter='\t')
           if self.ECGSeries.ndim > 1:
               self.ECGSeries = self.ECGSeries[:, 0]
           self.plot(self.ECGSeries[self.current_position:self.current_position+254],self.current_position)
        except Exception as e:
            print(f"Error al cargar o procesar el archivo: {e}")