import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from classification import Classifier
from multi_classification import MultiprocessClassifier

from tensorflow import keras
import h5py

'''
TEST 1: MNIST Tensor Flow classification test
Using the 28x28 TensorFlow set, compressed onto the range 0-1
'''
def MNISTTensorFlowTest(boundary, enc_zero=True):

    (X_train, y_train),(X_test, y_test) = \
        keras.datasets.mnist.load_data()
    
    X_train = X_train.reshape((X_train.shape[0], -1))
    y_train = y_train
    X_test = X_test.reshape((X_test.shape[0], -1))
    y_test = y_test

    X_train = 1*(X_train >= boundary)
    X_test = 1*(X_test >= boundary)
    
    if MULTI == 1:
        MNIST_classifier = Classifier()
    else:
        MNIST_classifier = MultiprocessClassifier(MULTI)
    MNIST_classifier.train(X_train, y_train, enc_zero=enc_zero)

    y_hat = MNIST_classifier.classify(X_test)

    print("\n\n")
    print(f"{X_train.shape[0]} training points")
    print(f"{X_test.shape[0]} testing points")
    print(f"{round(np.sum(y_hat == y_test) / y_hat.size, 2)} accuracy")
    print("\n\n")

'''
TEST 2: PCam data
Using the 96x96 PCam data focused onto the middle 32x32
Since the model takes a while to train, pickling is used for consistency
Note: Download the PCAM data from https://github.com/basveeling/pcam
'''
def PCamProcessing(boundary):

    X_train = h5py.File('camelyonpatch_level_2_split_train_x.h5', 'r')['x'][:1000]
    y_train = h5py.File('camelyonpatch_level_2_split_train_y.h5', 'r')['y'][:1000]
    X_test = h5py.File('camelyonpatch_level_2_split_test_x.h5', 'r')['x'][:200]
    y_test = h5py.File('camelyonpatch_level_2_split_test_y.h5', 'r')['y'][:200]

    X_train = np.array(X_train[:, 32:64, 32:64, :]).reshape((-1, 1024 * 3))
    y_train = np.array(y_train[:, 0, 0, 0])
    X_test = np.array(X_test[:, 32:64, 32:64, :]).reshape((-1, 1024 * 3))
    y_test = np.array(y_test[:, 0, 0, 0])

    X_train = 1*(X_train > boundary)
    X_test = 1*(X_test > boundary)

    if MULTI == 1:
        PCAM_classifier = Classifier()
    else:
        PCAM_classifier = MultiprocessClassifier(MULTI)
    PCAM_classifier.train(X_train, y_train)

    y_hat = PCAM_classifier.classify(X_test)

    print("\n\n")
    print(f"{X_train.shape[0]} training points")
    print(f"{X_test.shape[0]} testing points")
    print(f"{round(np.sum(y_hat == y_test) / y_hat.size, 2)} accuracy")
    print("\n\n")

'''
Running tests
'''
MULTI = 8 # Choose 1 for single classification, or higher for multiprocessing
if __name__ == "__main__":

    MNISTTensorFlowTest(85, enc_zero=False) # 7 runs - 0.75 accuracy
    # PCamProcessing(boundary = 150)