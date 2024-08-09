from datetime import date
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras import layers


from keras.layers import Input, Concatenate, Dense, Flatten, Dropout, Conv2D, BatchNormalization, GlobalMaxPooling2D

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNet

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tqdm.notebook import tqdm

parent_dir = os.path.split(os.getcwd())[0]
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts import dl_utils
from scripts import viz_tools

def scratch_original_model(input_shape):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2)),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2)),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(3)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model

def scratch_modified_model(input_shape):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            # layers.Conv2D(32, kernel_size=(3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2)),
            layers.Dropout(0.3),
    
            layers.Conv2D(64, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(64, kernel_size=(3), padding="same", activation="relu"),
            # layers.Conv2D(64, kernel_size=(3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2)),
            layers.Dropout(0.3),
    
            layers.Conv2D(128, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(128, kernel_size=(3), padding="same", activation="relu"),
            # layers.Conv2D(128, kernel_size=(3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2)),
            layers.Dropout(0.3),
    
            layers.Conv2D(256, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(512, kernel_size=(3), padding="same", activation="relu"),
            # layers.Conv2D(128, kernel_size=(3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(3)),
            layers.Dropout(0.3),
    
    
            # layers.Flatten(),
            GlobalMaxPooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    return model
    
def vgg_16_model(input_shape):
    # Importamos el modelo VGG16 preentrenado en ImageNet sin el top model
    base_model = VGG16(weights='imagenet',include_top=False)
    
    # Modificamos el input ya que VGG emplea imágenes RGB
    input_tensor = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    
    # Congelamos todas las capas del modelo base (la capa de input no se congela)
    for layer in base_model.layers:
        if 'block' not in layer.name or 'block5' in layer.name : #or 'block4' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    
    for layer in base_model.layers[2:]:
        x = layer(x)
    
    # Creamos un top model o clasificador
    x = Flatten()(x)
    x = Dense(1000, activation='relu', name='fc1')(x)
    x= Dropout(0.4)(x)
    x = Dense(200, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    
    model = Model(inputs=input_tensor, outputs=x)
    
    # Añadimos los pesos de las capas convolucionales de VGG16 al modelo
    for i in range(2, len(model.layers)):
        if isinstance(model.layers[i], Conv2D):
            # Obtener los pesos de la capa i en el modelo original
            original_weights = base_model.layers[i].get_weights()
            # Asignar esos pesos a la capa correspondiente en el nuevo modelo
            model.layers[i].set_weights(original_weights)
    
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model

def mobilenet_model(input_shape):
    # Importamos el modelo VGG16 preentrenado en ImageNet sin el top model
    base_model = MobileNet(weights='imagenet',include_top=False)
    
    # Modificamos el input ya que VGG emplea imágenes RGB
    input_tensor = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    
    for layer in base_model.layers[2:]:
        x = layer(x)
    
    # Creamos un top model o clasificador
    x = Flatten()(x)
    x = Dense(1000, activation='relu', name='fc1')(x)
    x= Dropout(0.4)(x)
    x = Dense(200, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    
    # Congelamos todas las capas del modelo base (la capa de input no se congela)
    for layer in base_model.layers:
        if ('conv_dw' not in layer.name and 'conv_pw' not in layer.name) :#or '_13' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    
    model = Model(inputs=input_tensor, outputs=x)
    
    # Añadimos los pesos de las capas convolucionales de VGG16 al modelo
    for i in range(2, len(model.layers)):
        if isinstance(model.layers[i], Conv2D):
            # Obtener los pesos de la capa i en el modelo original
            original_weights = base_model.layers[i].get_weights()
            # Asignar esos pesos a la capa correspondiente en el nuevo modelo
            model.layers[i].set_weights(original_weights)
    
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model