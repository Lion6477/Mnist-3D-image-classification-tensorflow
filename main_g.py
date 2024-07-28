# import numpy as np
# import h5py
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
#
# # Load data from HDF5 file
# def load_data_from_h5(file_path):
#     with h5py.File(file_path, 'r') as f:
#         X_train = np.array(f['X_train'])
#         y_train = np.array(f['y_train'])
#         X_test = np.array(f['X_test'])
#         y_test = np.array(f['y_test'])
#     return X_train, y_train, X_test, y_test
#
# # Preparation and training functions
# def prepare_data(file_path):
#     X_train, y_train, X_test, y_test = load_data_from_h5(file_path)
#
#     # Reshape data
#     X_train = X_train.reshape((-1, 16, 16, 16, 1))
#     X_test = X_test.reshape((-1, 16, 16, 16, 1))
#
#     # One-hot encoding
#     y_train = tf.keras.utils.to_categorical(y_train, 10)
#     y_test = tf.keras.utils.to_categorical(y_test, 10)
#
#     return X_train, y_train, X_test, y_test
#
# def train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=16, validation_split=0.2):
#     # Create the model
#     model = models.Sequential([
#         layers.Input(shape=(16, 16, 16, 1)),
#         layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
#         layers.MaxPooling3D((2, 2, 2)),
#         layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
#         layers.MaxPooling3D((2, 2, 2)),
#         layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
#         layers.MaxPooling3D((2, 2, 2)),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(10, activation='softmax')
#     ])
#
#     # Compile the model
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     # Train the model
#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
#
#     return model
#
# # Evaluation and result recording
# def evaluate_and_record(model, X_test, y_test, results_file='results.txt', append=True):
#     test_loss, test_acc = model.evaluate(X_test, y_test)
#
#     # Prepare results string
#     results_str = f"\nTest #{len(open(results_file, 'r').readlines()) + 1}:\n" \
#                   f"Parameters:\n" \
#                   f"  epochs: {epochs}\n" \
#                   f"  batch_size: {batch_size}\n" \
#                   f"  validation_split: {validation_split}\n" \
#                   f"Test accuracy: {test_acc}\n"
#
#     # Create the file if it doesn't exist (append mode handles existing files)
#     with open(results_file, 'a') as f:
#         f.write(results_str)  # Write results to the file
#
#     print(results_str)  # Print results for user feedback
#
# # Main execution
# if __name__ == '__main__':
#     file_path = 'full_dataset_vectors.h5'
#     X_train, y_train, X_test, y_test = prepare_data(file_path)
#
#     # Set training parameters (optional, adjust as needed)
#     epochs = 10
#     batch_size = 16
#     validation_split = 0.2
#
#     model = train_model(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
#     evaluate_and_record(model, X_test, y_test)
