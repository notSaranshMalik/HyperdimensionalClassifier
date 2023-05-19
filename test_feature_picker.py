from tensorflow import keras
from feature_picker import FeaturePicker
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py
import numpy as np

# Data import

# (X_train, y_train),(X_test, y_test) = \
#         keras.datasets.mnist.load_data()
# X_train = X_train.reshape(X_train.shape[0], -1)
# X_test = X_train.reshape(X_train.shape[0], -1)

X_train = h5py.File('camelyonpatch_level_2_split_train_x.h5', 'r')['x']
y_train = h5py.File('camelyonpatch_level_2_split_train_y.h5', 'r')['y']
X_test = h5py.File('camelyonpatch_level_2_split_test_x.h5', 'r')['x']
y_test = h5py.File('camelyonpatch_level_2_split_test_y.h5', 'r')['y']
X_train = np.array(X_train[:, 32:64, 32:64, :]).reshape((-1, 1024 * 3))
y_train = np.array(y_train[:, 0, 0, 0])
X_test = np.array(X_test[:, 32:64, 32:64, :]).reshape((-1, 1024 * 3))
y_test = np.array(y_test[:, 0, 0, 0])

# Splitting off a validation set for feature picking
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1)

'''
TEST 1: Matrix test
'''
def test_matrix():
    classes = FeaturePicker.calculateMatrix(X_val, y_val)
    for val in classes.values():
        plt.matshow(val.reshape((28, 28)))
        plt.show()

'''
TEST 2: Importance test
'''
def test_importance():
    importance = FeaturePicker.pickFeatures(X_val, y_val)
    plt.matshow(importance.reshape((28, 28)))
    plt.show()

'''
Running tests
'''
if __name__ == "__main__":
    test_matrix()
    # test_importance()