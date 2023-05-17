import numpy as np
from classification import Classifier
from tqdm import tqdm

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow import keras
import h5py
import pickle
'''
TEST 1: MNIST classification test
Using the 8x8 SKlearn set
'''
def MNISTTest():

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)
    
    print("\n\nBEGIN TRAINING")
    MNIST_classifier = Classifier()
    MNIST_classifier.train(X_train, y_train)

    print("\n\nBEGIN TESTING")
    y_hat = MNIST_classifier.classify(X_test)

    print("\n\n")
    print(f"{X_train.shape[0]} training points")
    print(f"{X_test.shape[0]} testing points")
    print(f"{round(np.sum(y_hat == y_test) / y_hat.size, 2)} accuracy")
    print("\n\n")

'''
TEST 2: MNIST compressed classification test
Using the 8x8 SKlearn set - compressed onto the range 0-1
'''
def MNISTTestCompress(boundary, enc_zero=True):

    X, y = load_digits(return_X_y=True)
    X = 1 * (X >= boundary)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)
    
    print("\n\nBEGIN TRAINING")
    MNIST_classifier = Classifier()
    MNIST_classifier.train(X_train, y_train, enc_zero=enc_zero)

    print("\n\nBEGIN TESTING")
    y_hat = MNIST_classifier.classify(X_test)

    print("\n\n")
    print(f"{X_train.shape[0]} training points")
    print(f"{X_test.shape[0]} testing points")
    print(f"{round(np.sum(y_hat == y_test) / y_hat.size, 2)} accuracy")
    print("\n\n")


'''
TEST 3: MNIST Tensor Flow classification test
Using the 28x28 TensorFlow set, compressed onto the range 0-1
'''
def MNISTTensorFlowTest(boundary, enc_zero=True):

    (X_train, y_train),(X_test, y_test) = \
        keras.datasets.mnist.load_data()
    
    X_train = 1*(X_train >= boundary).reshape((X_train.shape[0], -1))
    y_train = y_train
    X_test = 1*(X_test >= boundary).reshape((X_test.shape[0], -1))
    y_test = y_test
    
    print("\n\nBEGIN TRAINING")
    MNIST_classifier = Classifier()
    MNIST_classifier.train(X_train, y_train, enc_zero=enc_zero)

    print("\n\nBEGIN TESTING")
    y_hat = MNIST_classifier.classify(X_test)

    print("\n\n")
    print(f"{X_train.shape[0]} training points")
    print(f"{X_test.shape[0]} testing points")
    print(f"{round(np.sum(y_hat == y_test) / y_hat.size, 2)} accuracy")
    print("\n\n")

'''
TEST 4: PCam data
Using the 96x96 PCam data focused onto the middle 32x32
Since the model takes a while to train, pickling is used for consistency
Note: Download the PCAM data from https://github.com/basveeling/pcam
'''
def PCamProcessing():

    print("\n\nLOADING DATA")
    x_train = h5py.File('camelyonpatch_level_2_split_train_x.h5', 'r')['x'][:1000]
    y_train = h5py.File('camelyonpatch_level_2_split_train_y.h5', 'r')['y'][:1000]

    x_test = h5py.File('camelyonpatch_level_2_split_test_x.h5', 'r')['x'][:200]
    y_test = h5py.File('camelyonpatch_level_2_split_test_y.h5', 'r')['y'][:200]

    x_train = np.array(x_train[:, 32:64, 32:64, :]).reshape((-1, 1024, 3))
    y_train = np.array(y_train[:, 0, 0, 0])

    x_test = np.array(x_test[:, 32:64, 32:64, :]).reshape((-1, 1024, 3))
    y_test = np.array(y_test[:, 0, 0, 0])

    print("\n\nPROCESSING TRAINING DATA")
    x_train_quant = np.empty((x_train.shape[0], 1024))
    for row in tqdm(range(x_train.shape[0])):
        for col in range(1024):
            x_train_quant[row][col] = np.sum(x_train[row][col])

    print("\n\nPROCESSING TESTING DATA")
    x_test_quant = np.empty((x_test.shape[0], 1024))
    for row in tqdm(range(x_test.shape[0])):
        for col in range(1024):
            x_test_quant[row][col] = np.sum(x_test[row][col])

    with open('processed_x_train.pkl', 'wb') as f:
        pickle.dump(x_train_quant, f)

    with open('processed_y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)

    with open('processed_x_test.pkl', 'wb') as f:
        pickle.dump(x_test_quant, f)

    with open('processed_y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

def PCamTest(boundary, enc_zero=True):

    with open('processed_x_train.pkl', 'rb') as f:
        X_train = pickle.load(f)

    with open('processed_y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)

    with open('processed_x_test.pkl', 'rb') as f:
        X_test = pickle.load(f)

    with open('processed_y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    X_train = 1*(X_train > boundary)
    X_test = 1*(X_test > boundary)

    print("\n\nBEGIN TRAINING")
    PCAM_classifier = Classifier()
    PCAM_classifier.train(X_train, y_train, enc_zero=enc_zero)

    print("\n\nBEGIN TESTING")
    y_hat = PCAM_classifier.classify(X_test)

    print("\n\n")
    print(f"{X_train.shape[0]} training points")
    print(f"{X_test.shape[0]} testing points")
    print(f"{round(np.sum(y_hat == y_test) / y_hat.size, 2)} accuracy")
    print("\n\n")


'''
Running tests
'''
if __name__ == "__main__":
    # MNISTTest() # 8 runs - 0.76 average accuracy
    # MNISTTestCompress(8) # 4 runs - 0.84 accuracy
    # MNISTTestCompress(8, enc_zero=False) # 2 runs - 0.85 average accuracy
    # MNISTTestCompress(1) # 2 runs - 0.84 average accuracy
    # MNISTTestCompress(1, enc_zero=False) # 2 runs - 0.78 average accuracy
    MNISTTensorFlowTest(85, enc_zero=False) # 2 runs - 0.76 accuracy
    
    # PCamProcessing()
    # PCamTest(600, enc_zero=False)