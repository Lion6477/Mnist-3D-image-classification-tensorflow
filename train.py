import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

from utils import load_data_from_h5

# Параметры обучения
epochs = 8
learning_rate = 0.1
batch_size = 8

# Подготовка данных
file_path = 'full_dataset_vectors.h5'
X_train, y_train, X_test, y_test = load_data_from_h5(file_path)

# Reshaping data in form (num_samples, 16, 16, 16, 1)
X_train = X_train.reshape((-1, 16, 16, 16, 1))
X_test = X_test.reshape((-1, 16, 16, 16, 1))

# One-hot coding for labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Creating model
model = models.Sequential()

model.add(layers.Input(shape=(16, 16, 16, 1)))
model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling3D((2, 2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Model compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model training
# Train the model with batch size
# history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)
# model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Train the model with batch size
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size)

# Получение финальных значений loss и accuracy
final_loss, final_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Training completed")
name = ("trained_model" +
        f"_epochs-{epochs}" +
        f"_rate-{learning_rate}" +
        "_loss-" + str(round(final_loss, 3)) +
        "_acc-" + str(round(final_accuracy, 3)) + ".keras")
print(f"Save model into file \"{name}\"? y/n")

if input().lower() == "n":
    print("Model not saved")

else:
    print(f"Saving as {name}")
    model.save(name)
    print("Model saved")