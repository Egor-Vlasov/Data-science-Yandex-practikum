from keras.models import Sequential
import numpy as np
from keras.layers import  GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization,Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50
from keras.regularizers import l2
import pandas as pd


def load_train(path):
    
    labels = pd.read_csv(path + 'labels.csv')

    datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1/255.,
        horizontal_flip=True,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.1,
    )

    return datagen.flow_from_dataframe(
        dataframe = labels,
        directory = path+'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=123,
    )


def load_test(path):
    
    labels = pd.read_csv(path + 'labels.csv')

    datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1/255.)

    return datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path+'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=123,
    )


    



def create_model(input_shape=(224, 224, 3)):
    """Создание модели на основе ResNet50"""
    print("Модель нейросети в процессе подготовки...")
    
    weights_path = '/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # Загрузка предобученной ResNet50 без верхних слоев
    backbone = ResNet50(
        input_shape=input_shape,
        weights=weights_path,  # Используем стандартные веса imagenet
        include_top=False
    )
    
    
    # for layer in backbone.layers[:100]:
    #     layer.trainable = False
    # for layer in backbone.layers[100:]:
    #     layer.trainable = True
    
    model = Sequential([
        backbone,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        # Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        # Dropout(0.5),
        Dense(1) 
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  
        loss='mse',  # MSE для регрессии
        metrics=['mae']  # MAE как метрика
    )
    
    print("Модель нейросети готова!")
    return model

def train_model(model, train_data, val_data, epochs=40, steps_per_epoch=None, validation_steps=None):
   
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2
    )
    
    return model
