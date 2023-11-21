#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 13:35:17 2023

@author: ivanlorenzanabelli
"""

import sys
from PyQt5.QtWidgets import QApplication
from app import App

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
