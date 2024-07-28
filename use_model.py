import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Функция для предсказания класса на основе входных данных
def predict(model, data):
    data = data.reshape((-1, 16, 16, 16, 1))  # Ensure data is reshaped correctly
    prediction = model.predict(data)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

# Загрузка обученной модели
# model_name = input("Введите имя модели для использования: ")
model_name = "trained_model_epochs-10_rate-0.01_loss-1.141_acc-0.747.keras"
model = load_model(model_name)

# Пример использования
# Загрузка примера данных для предсказания (замените этот код на ваш источник данных)
example_data = np.random.rand(1, 16, 16, 16)
#TODO: here is data 3D-image for use model

predicted_class = predict(model, example_data)
print(f"Predicted Class: {predicted_class[0]}")
