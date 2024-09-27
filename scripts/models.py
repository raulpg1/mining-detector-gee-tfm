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

import tensorflow as tf

from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, GlobalMaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNet

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

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
            layers.MaxPooling2D(pool_size=(2)),
            layers.Dropout(0.3),
    
            layers.Conv2D(64, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(64, kernel_size=(3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2)),
            layers.Dropout(0.3),
    
            layers.Conv2D(128, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(128, kernel_size=(3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2)),
            layers.Dropout(0.3),
    
            layers.Conv2D(256, kernel_size=(3), padding="same", activation="relu"),
            layers.Conv2D(512, kernel_size=(3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(3)),
            layers.Dropout(0.3),
            
            layers.GlobalMaxPooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
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
    
def vgg_16_model(input_shape,top_model_layer):
    # Importamos el modelo VGG16 preentrenado en ImageNet sin el top model
    base_model = VGG16(weights='imagenet',include_top=False)
    
    # Modificamos el input ya que VGG emplea imágenes RGB
    input_tensor = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    
    # Congelamos todas las capas del modelo base exceptuando los primeros bloques convolucionales
    entrenable = True
    for layer in base_model.layers:
        if layer.name == "block3_conv1": # Descongelamos los 3 primeros bloques ya que se ha modificado el input
            entrenable = False
        if layer.name == "block5_conv1": # Descongelamos la última capa ya que nuestro dataset no se parece a ImageNet
            entrenable = True
        layer.trainable = entrenable
    
    for layer in base_model.layers[2:]:
        x = layer(x)

    
    # Creamos un top model o clasificador
    if top_model_layer == "Flatten":
        x = Flatten()(x)
    elif top_model_layer == "GlobalMaxPooling":
        x = GlobalMaxPooling2D()(x)
    elif top_model_layer == "GlobalAveragePooling":
        x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
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

def mobilenet_model(input_shape,top_model_layer):
    base_model = MobileNet(weights='imagenet',include_top=False)
    
    input_tensor = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    
    for layer in base_model.layers[2:]:
        x = layer(x)
    
    # Creamos un top model o clasificador
    # Seleccionamos una de las siguientes capas
    if top_model_layer == "Flatten":
        x = Flatten()(x)
    elif top_model_layer == "GlobalMaxPooling":
        x = GlobalMaxPooling2D()(x)
    elif top_model_layer == "GlobalAveragePooling":
        x = GlobalAveragePooling2D()(x)
        
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    
    # # Congelamos todas las capas del modelo a partir del bloque 6
    entrenable = True
    for layer in base_model.layers:
        if layer.name == "conv_pad_4": # Descongelamos los 3 primeros bloques ya que se ha modificado el input
            entrenable = False
        if layer.name == "conv_dw_11": # Descongelamos las últimas capas ya que nuestro dataset no se parece a ImageNet
            entrenable = True
        layer.trainable = entrenable
    
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

def alexnet_model(input_shape):
    
    model = keras.Sequential(
        [
            layers.Conv2D(96, (5, 5), strides=1, activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    return model 
