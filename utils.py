import h5py

# Load data from HDF5 file
def load_data_from_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        X_train = f["X_train"][:]
        y_train = f["y_train"][:]
        X_test = f["X_test"][:]
        y_test = f["y_test"][:]
    return X_train, y_train, X_test, y_test
