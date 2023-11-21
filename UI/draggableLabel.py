#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 13:33:17 2023

@author: ivanlorenzanabelli
"""
from PyQt5.QtWidgets import QLabel

class DraggableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()  
        self.parent().handleFile(file_path)  # Llama al método handleFile de la clase principal