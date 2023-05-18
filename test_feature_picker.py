from tensorflow import keras
from feature_picker import FeaturePicker
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Data import
(X_train, y_train),(X_test, y_test) = \
        keras.datasets.mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_train.reshape(X_train.shape[0], -1)

# Splitting off a validation set for feature picking
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1)

'''
TEST 1: Matrix test
'''
def test_matrix():
    classes = FeaturePicker.calculate_matrix(X_val, y_val)
    for val in classes.values():
        plt.matshow(val.reshape((28, 28)))
        plt.show()

'''
TEST 2: Importance test
'''
def test_importance():
    importance = FeaturePicker.pick_features(X_val, y_val)
    plt.matshow(importance.reshape((28, 28)))
    plt.show()

'''
Running tests
'''
if __name__ == "__main__":
    # test_matrix()
    test_importance()