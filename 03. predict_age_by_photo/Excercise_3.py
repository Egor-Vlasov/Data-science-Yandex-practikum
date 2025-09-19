
# Задание 3
# Постройте и обучите свёрточную нейронную сеть на наборе данных с фруктами. Для этого создайте в коде три функции:
# загрузки обучающей выборки load_train() (теперь функция вернёт загрузчик данных),
# создания модели create_model(),
# запуска модели train_model().
# Добейтесь того, чтобы значение accuracy на тестовой выборке было не меньше 90%.
# У вас есть ограничение: модель должна обучиться за час.
# Для выполнения задания будет использована полная версия датасета с фруктами. Путь к ней уже указан в прекоде.



from keras.models import Sequential
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_train(path='/datasets/fruits_small/'):
    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1/255.
    )
    
    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,  
        class_mode='sparse',
        subset='training',
        seed=12345
    )
    return train_generator

def create_model(input_shape=(150, 150, 3)):
    
    model = Sequential([
        # Первый слой
        Conv2D(8,(3, 3), activation='relu',padding ='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        #второй слой
        Conv2D(16, (3, 3), activation='relu',padding ='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        #третий слой
        Conv2D(32, (3, 3), activation='relu',padding ='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        #четвертый слой
        Conv2D(64, (3, 3), activation='relu',padding ='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        #пятый слой
        Conv2D(128, (3, 3), activation='relu',padding ='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        # Полносвязные слои
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(12, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model




def train_model(model, train_generator, val_generator, batch_size=None, epochs=2, 
               steps_per_epoch=None, validation_steps=None):
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2
    )
    return model
