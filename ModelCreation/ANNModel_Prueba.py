import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adamax, Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1, l2
from imblearn.over_sampling import SMOTE

# Cargar datos
rutaR_csv = 'C:\\Users\\XPG\\Desktop\\DiagnosticoAsistido\\Arrhythmia-Detector\\Data\\Nueva carpeta\\combined_ecg_R.csv'  # Reemplaza con la ruta a tu archivo CSV
rutaB_csv = 'C:\\Users\\XPG\\Desktop\\DiagnosticoAsistido\\Arrhythmia-Detector\\Data\\Nueva carpeta\\combined_ecg_B.csv'  # Reemplaza con la ruta a tu archivo CSV
datosR = pd.read_csv(rutaR_csv)
datosB = pd.read_csv(rutaB_csv)

# Separar características y etiquetas
X_r = datosR[['WaveletEnergy', 'ShannonEntropy', 'Kurtosis', 'Variance']]
X_b = datosB[['WaveletEnergy', 'SignalSTD', 'Kurtosis']]

y_rhythm = to_categorical(datosR['RhythmClass'])
y_beat = to_categorical(datosB['BeatType'])

# Dividir en conjuntos de entrenamiento y prueba para ritmo
X_train_r, X_test_r, y_train_rhythm, y_test_rhythm = train_test_split(X_r, y_rhythm, test_size=0.3, random_state=42)

# Dividir en conjuntos de entrenamiento y prueba para tipo de latido
X_train_b, X_test_b, y_train_beat, y_test_beat = train_test_split(X_b, y_beat, test_size=0.3, random_state=42)


# Normalizar características
scaler = StandardScaler()
X_Rtrain = scaler.fit_transform(X_train_r)
X_Rtest = scaler.transform(X_test_r)

X_Btrain = scaler.fit_transform(X_train_b)
X_Btest = scaler.transform(X_test_b)

# Aplicar SMOTE
smote = SMOTE()
# Aplicar SMOTE para el tipo de latido
X_train_smote_beat, y_train_beat_smote = smote.fit_resample(X_train_b, y_train_beat.argmax(axis=1))
y_train_beat_smote = to_categorical(y_train_beat_smote)

# Aplicar SMOTE para la clase de ritmo
X_train_smote_rhythm, y_train_rhythm_smote = smote.fit_resample(X_train_r, y_train_rhythm.argmax(axis=1))
y_train_rhythm_smote = to_categorical(y_train_rhythm_smote)



# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

model_beat = Sequential([
    Dense(1024, input_shape=(X_train_b.shape[1],), activation='leaky_relu', kernel_regularizer=l2(0.015)),
    BatchNormalization(),
    Dropout(0.7),
    Dense(1024, activation='leaky_relu', kernel_regularizer=l1(0.01)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(512, activation='leaky_relu', kernel_regularizer=l1(0.01)),
    Dense(y_train_beat.shape[1], activation='softmax')
])

optimizer_sgd_beat = SGD(learning_rate=0.05, momentum=0.9)
model_beat.compile(optimizer=optimizer_sgd_beat, 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])
# optimizer = Adam(learning_rate=0.01)
# model_beat.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo para el tipo de latido con los datos balanceados
history_beat = model_beat.fit(X_train_smote_beat, y_train_beat_smote, 
                              validation_split=0.15, epochs=90, 
                              batch_size=128, shuffle=True, verbose=1, 
                              callbacks=[early_stopping, reduce_lr])
model_beat.summary()
model_beat.get_weights()
model_beat.optimizer

model_rhythm = Sequential([
    Dense(1024, input_shape=(X_train_r.shape[1],), activation='leaky_relu', kernel_regularizer=l2(0.015)),
    BatchNormalization(),
    Dropout(0.7),
    Dense(1024, activation='leaky_relu', kernel_regularizer=l1(0.01)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(512, activation='leaky_relu', kernel_regularizer=l1(0.01)),
    Dense(y_train_rhythm.shape[1], activation='softmax')
])

# Compilar el modelo para la clase de ritmo
# optimizer_adamax_beat = Adamax(learning_rate=0.005)
# model_rhythm.compile(optimizer=optimizer_adamax_beat, 
#                    loss='categorical_crossentropy', 
#                    metrics=['accuracy'])
# optimizer_sgd_beat = SGD(learning_rate=0.05, momentum=0.9)
# model_rhythm.compile(optimizer=optimizer_sgd_beat, 
#                    loss='categorical_crossentropy', 
#                    metrics=['accuracy'])
# optimizer_rmsprop = RMSprop(learning_rate=0.03)
# model_rhythm.compile(optimizer=optimizer_rmsprop, 
#                      loss='categorical_crossentropy', 
#                      metrics=['accuracy'])
optimizer = Adam(learning_rate=0.01)
model_rhythm.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])# 

# Entrenar el modelo para la clase de ritmo
history_rhythm = model_rhythm.fit(X_train_smote_rhythm, y_train_rhythm_smote, 
                                  validation_split=0.15, epochs=90, 
                                  batch_size=90, shuffle=True, verbose=1, 
                                  callbacks=[early_stopping, reduce_lr])
model_rhythm.summary()
model_rhythm.get_weights()
model_rhythm.optimizer

# Evaluar el modelo para el tipo de latido
beat_loss, beat_accuracy = model_beat.evaluate(X_test_b, y_test_beat)
print(f'Beat Model - Loss: {beat_loss}, Accuracy: {beat_accuracy}')

# Evaluar el modelo para la clase de ritmo
rhythm_loss, rhythm_accuracy = model_rhythm.evaluate(X_test_r, y_test_rhythm)
print(f'Rhythm Model - Loss: {rhythm_loss}, Accuracy: {rhythm_accuracy}')

# Guardar los modelos
model_beat.save('C:\\Users\\XPG\\Desktop\\DiagnosticoAsistido\\Arrhythmia-Detector\\Models\\modelo_beat.h5')
model_rhythm.save('C:\\Users\\XPG\\Desktop\\DiagnosticoAsistido\\Arrhythmia-Detector\\Models\\modelo_rhythm.h5')
print("Modelos guardados como 'modelo_beat.h5' y 'modelo_rhythm.h5'")
joblib.dump(scaler, 'C:\\Users\\XPG\\Desktop\\DiagnosticoAsistido\\Arrhythmia-Detector\\Models\\scaler_ecg.pk1')

# Predicciones y Matrices de Confusión
y_pred_beat = model_beat.predict(X_test_b)
y_pred_rhythm = model_rhythm.predict(X_test_r)
y_pred_beat = np.argmax(y_pred_beat, axis=1)
y_pred_rhythm = np.argmax(y_pred_rhythm, axis=1)

# Matriz de confusión y reporte de clasificación para el tipo de latido
print("Confusion Matrix for Beat Type:")
cm_beat = confusion_matrix(y_test_beat.argmax(axis=1), y_pred_beat)
print(cm_beat)
print("Classification Report for Beat Type:")
print(classification_report(y_test_beat.argmax(axis=1), y_pred_beat, zero_division=1))

# Matriz de confusión y reporte de clasificación para la clase de ritmo
print("Confusion Matrix for Rhythm Class:")
cm_rhythm = confusion_matrix(y_test_rhythm.argmax(axis=1), y_pred_rhythm)
print(cm_rhythm)
print("Classification Report for Rhythm Class:")
print(classification_report(y_test_rhythm.argmax(axis=1), y_pred_rhythm))

# Gráficas de rendimiento para el modelo de tipo de latido
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_beat.history['accuracy'])
plt.plot(history_beat.history['val_accuracy'])
plt.title('Beat Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Gráficas de rendimiento para el modelo de clase de ritmo
plt.subplot(1, 2, 2)
plt.plot(history_rhythm.history['accuracy'])
plt.plot(history_rhythm.history['val_accuracy'])
plt.title('Rhythm Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Función para graficar la matriz de confusión
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Nombres de las clases para los beats y ritmos
class_names_beat = ['Normal', 'Left bundle branch block', 'Right bundle branch block', 'Atrial premature','Premature ventricular contraction', 'Fusion of ventricular and normal beat']
class_names_rhythm = ['Normal', 'Bradycardia', 'Tachycardia'] 
# Graficar las matrices de confusión para ambos modelos
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_beat, class_names_beat, title='Confusion Matrix for Beat Type')
plt.show()

plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_rhythm, class_names_rhythm, title='Confusion Matrix for Rhythm Class')
plt.show()
