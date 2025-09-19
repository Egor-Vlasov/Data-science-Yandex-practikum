
# Задание 1
# Постройте и обучите нейронную сеть на наборе данных с предметами одежды. Для этого создайте в коде три функции:
# загрузки обучающей выборки load_train(),
# создания модели create_model(),
# запуска модели train_model().
# Добейтесь того, чтобы значение accuracy на тестовой выборке было не меньше 85%.





from keras.datasets import fashion_mnist
from keras.models import Sequential
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

def load_train(path):
    """
    Загружает и подготавливает тренировочные данные
    Args:
        path (str): Путь к директории с файлами данных
    Returns:
        tuple: (features_train, target_train) - тренировочные данные и метки
    """
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    # Преобразуем в 4D-тензор (N, 28, 28, 1) и нормализуем [0, 1]
    features_train = features_train.reshape(-1, 28, 28, 1) / 255.
    return features_train, target_train

def create_model(input_shape=(28, 28, 1)):
    """
    Создаёт модель CNN
    Args:
        input_shape (tuple): Форма входных данных (по умолчанию (28, 28, 1))
    Returns:
        Sequential: Готовая к обучению модель
    """
    model = Sequential([
        # Первый свёрточный слой
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Второй свёрточный слой
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Третий свёрточный слой
        Conv2D(128, (3, 3), activation='relu'),
        
        # Полносвязные слои
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(model, train_data, test_data, batch_size=32, epochs=10,
               steps_per_epoch=None, validation_steps=None):
    """
    Обучает модель на предоставленных данных
    Args:
        model: Модель для обучения
        train_data: Тренировочные данные
        test_data: Тестовые данные
        batch_size: Размер батча
        epochs: Количество эпох
    Returns:
        History: История обучения
    """
    features_train, target_train = train_data
    features_test, target_test = test_data
    
    history = model.fit(features_train, target_train,
                        validation_data=(features_test, target_test),
                        batch_size=batch_size,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        verbose=2,
                        shuffle=True)
    
    return model