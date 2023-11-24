# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 23:17:00 2019

@author: papacito
"""
import numpy as np

__author__ = "Arturo Sotelo-Orozco"
__version__ = "1.0.0"
__license__ = "TecNM-ITT"

#N number of samples
#A1 component 1 amplitude (V)
#A2 component 2 amplitude (V) 
#f1 component 1 frequency (Hz)
#f2 component 2 frequency (Hz)
#pashe1 component 1 phase (degrees)
#pashe2 component 2 phase (degrees)

# =============================================================================
# Sine wave two components
# =============================================================================
def sineWave(N, A1, A2, f1, f2, fs, phase1, phase2):

    T = 1.0/fs    
#Components settings    
    ph1 = phase1*np.pi/180.0 #Fundamental component phase
    ph2 = phase2*np.pi/180.0 #third harmonic phase 
    
#building componentes
    sDuration = N*T #signal duration
    t = np.linspace(0.0, sDuration,N) #bin from 0s to 0.5s over 600 samples
    c1 = A1*np.sin((f1 * 2.0 * np.pi* t) + ph1)
    c2 = A2*np.sin((f2 * 2.0 * np.pi* t) + ph2)

#Building synthetic signal
    y = c1 + c2
    
    return y, t

# =============================================================================
# Sine wave three components
# =============================================================================
def sineWave3c(N, A1, A2, A3, f1, f2, f3, fs, phase1, phase2, phase3):

    T = 1.0/fs   
#Components settings    
    ph1 = phase1*np.pi/180.0 #Fundamental component phase
    ph2 = phase2*np.pi/180.0 #third harmonic phase 
    ph3 = phase3*np.pi/180.0 #third harmonic phase
    
#building componentes
    sDuration = N*T #signal duration
    t = np.linspace(0.0, sDuration,N) #bin from 0s to 0.5s over 600 samples
    c1 = A1*np.sin((2.0 * np.pi* f1 *  t) + ph1)
    c2 = A2*np.sin((2.0 * np.pi* f2 *  t) + ph2)
    c3 = A3*np.sin((2.0 * np.pi* f3 *  t) + ph3)

#Building synthetic signal
    y = c1 + c2 + c3
    return y, t

# =============================================================================
# White Gaussian noise
# =============================================================================
def awgnoise(N, An, fs):
    #Withe Gaussian Noise generation    
    T = 1.0/fs    
    sDuration = N*T #signal duration
    t = np.linspace(0.0, sDuration,N) #bin from 0s to 0.5s over 600 samples
    awgn=np.random.randn(N)
    awgn= An * awgn/np.amax(awgn)  #Adding offset and modfying the amplitude

    return awgn, t

# =============================================================================
# Generate a test signal, an cA Vp sine wave whose   amplitud cA is 
# slowly modulated around an information signal of iA Vp and iF Hz, corrupted
# by WGN of exponentially decreasing magnitude sampled at fs Hz.
# =============================================================================
def AMsignal(N, fs, cA, cF, iA, iF, nA):
#    fs = 10e3 #Hz
    Ts = 1/fs
#    N = 1e5 #Samples
    time = np.arange(N)* Ts
    mod = iA*np.cos(2*np.pi*iF*time)
    carrier = cA*(1 + mod)* np.sin(2*np.pi*cF*time )
    noise = np.random.normal(scale=nA, size=time.shape)
    x = carrier + noise
    
    return  x, time, carrier, mod, noise

# =============================================================================
# Generate a test signal, an cA Vp sine wave whose frequency of cF Hz is 
# slowly modulated around an information signal of iA Vp and iF Hz, corrupted
# by WGN of exponentially decreasing magnitude sampled at fs Hz.
# =============================================================================
def FMsignal(N, fs, cA, cF, iA, iF, nA):
#    fs = 10e3 #Hz
    Ts = 1/fs
#    N = 1e5 #Samples
    time = np.arange(N)* Ts
    mod = iA*np.cos(2*np.pi*iF*time)
    carrier = cA * np.sin(2*np.pi*cF*time + mod)
    noise = np.random.normal(scale=nA, size=time.shape)
    x = carrier + noise
    
    return  x, time, carrier, mod, noise

# =============================================================================
# Rectangular window
# =============================================================================
def boxcar(boxLen, sigLen, leadingZeros):
    leadZeros = np.zeros(leadingZeros)
    boxWin = np.ones(boxLen)
    trailingZ = np.zeros(sigLen-(leadingZeros+boxLen))
    boxWin = np.append(leadZeros, boxWin)
    boxWin = np.append(boxWin, trailingZ )
    winName = 'Boxcar'

    return boxWin, winName

# =============================================================================
# Hamming window
# =============================================================================
def hamming(boxLen, sigLen, leadingZeros):
    leadZeros = np.zeros(leadingZeros)
    boxWin = np.hamming(boxLen)
    trailingZ = np.zeros(sigLen-(leadingZeros+boxLen))
    boxWin = np.append(leadZeros, boxWin)
    boxWin = np.append(boxWin, trailingZ )
    winName = 'Hamming'

    return boxWin, winName

# =============================================================================
# Hanning window
# =============================================================================
def hanning(boxLen, sigLen, leadingZeros):
    leadZeros = np.zeros(leadingZeros)
    boxWin = np.hanning(boxLen)
    trailingZ = np.zeros(sigLen-(leadingZeros+boxLen))
    boxWin = np.append(leadZeros, boxWin)
    boxWin = np.append(boxWin, trailingZ )
    winName = 'Hanning'

    return boxWin, winName

# =============================================================================
# Blacman window
# =============================================================================
def blackman(boxLen, sigLen, leadingZeros):
    leadZeros = np.zeros(leadingZeros)
    boxWin = np.blackman(boxLen)
    trailingZ = np.zeros(sigLen-(leadingZeros+boxLen))
    boxWin = np.append(leadZeros, boxWin)
    boxWin = np.append(boxWin, trailingZ )
    winName = 'Blackman'

    return boxWin, winName
