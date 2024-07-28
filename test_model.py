import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from utils import load_data_from_h5

# Подготовка данных
file_path = 'full_dataset_vectors.h5'
X_train, y_train, X_test, y_test = load_data_from_h5(file_path)

# Reshaping data in form (num_samples, 16, 16, 16, 1)
X_test = X_test.reshape((-1, 16, 16, 16, 1))

# One-hot coding for labels
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Загрузка обученной модели
# model_name = input("Введите имя модели для тестирования: ")
model_name = "trained_model_epochs-10_rate-0.01_loss-1.141_acc-0.747.keras"
model = load_model(model_name)

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Предсказания
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Отчет по классификации
report = classification_report(y_true_classes, y_pred_classes)
print("Classification Report:")
print(report)
