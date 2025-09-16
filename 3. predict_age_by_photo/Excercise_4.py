from keras.models import Sequential
import numpy as np
from keras.layers import  GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization,Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50


def load_train(path='/datasets/fruits_small/'):
    print("тренировочные данные в процессе подготовки...")
    train_datagen = ImageDataGenerator(
    #    validation_split=0.25,
        rescale=1/255.
    )
    
    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=64,  
        class_mode='sparse',
    #    subset='training',
        seed=12345
    )
    print("Тренировачные данные готовы!")
    return train_generator

def create_model(input_shape=(150, 150, 3)):

    print("Модель нейросети в процессе подготовки...")
    weights_path = '/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    backbone = ResNet50(
        input_shape=input_shape,
        weights=weights_path,
        include_top=False  # Не включаем верхние слои (классификатор)
    )

    backbone.trainable = True

    model = Sequential([
        backbone,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(12, activation='softmax')
    ])    
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Модель нейросети готова!")
    return model



def train_model(model, train_data, test_data, batch_size=None, epochs=10, 
               steps_per_epoch=None, validation_steps=None):
    print("Процесс обучения...")
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2
    )
    print("Модель обучена!")
    return model
