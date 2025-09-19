
# Задание 4
# Постройте и обучите архитектуру ResNet на наборе данных с фруктами. Добейтесь того, чтобы значение accuracy на тестовой выборке было не меньше 99%.
# У вас есть ограничение: модель должна обучиться за полчаса.



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


# Вывод
#  - 308s - loss: 0.1016 - accuracy: 0.9706 - val_loss: 45.2298 - val_accuracy: 0.0696
# Epoch 2/10
#  - 147s - loss: 0.0400 - accuracy: 0.9886 - val_loss: 3.5213 - val_accuracy: 0.1066
# Epoch 3/10
#  - 149s - loss: 0.0153 - accuracy: 0.9951 - val_loss: 1.4577 - val_accuracy: 0.7191
# Epoch 4/10
#  - 150s - loss: 0.0235 - accuracy: 0.9933 - val_loss: 8.8804e-04 - val_accuracy: 0.9896
# Epoch 5/10
#  - 149s - loss: 0.0276 - accuracy: 0.9921 - val_loss: 2.3788 - val_accuracy: 0.6808
# Epoch 6/10
#  - 148s - loss: 0.0098 - accuracy: 0.9973 - val_loss: 0.0709 - val_accuracy: 0.9947
# Epoch 7/10
#  - 150s - loss: 0.0147 - accuracy: 0.9956 - val_loss: 0.4094 - val_accuracy: 0.9085
# Epoch 8/10
#  - 149s - loss: 0.0052 - accuracy: 0.9985 - val_loss: 0.2787 - val_accuracy: 0.9557
# Epoch 9/10
#  - 149s - loss: 0.0223 - accuracy: 0.9932 - val_loss: 0.1621 - val_accuracy: 0.7722
# Epoch 10/10
#  - 148s - loss: 0.0112 - accuracy: 0.9964 - val_loss: 9.8540e-05 - val_accuracy: 0.9992
# Модель обучена!