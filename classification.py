import warnings
warnings.filterwarnings("ignore")

from vector import Vector
from vector_space import VectorSpace
from vector_groups import VectorGroups

import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from copy import deepcopy
from os import cpu_count
from sklearn.model_selection import train_test_split
from tqdm.gui import tqdm

SIZE = 10000

def encode(x, features, values):
    '''
    Encodes a data point to it's hyperdimensional vector
    '''
    sum = Vector(SIZE, zero_vec=True)
    for j in range(x.size):
        sum += features[j] * values[x[j]]
    return sum

def train(X, y, features, values, classes):
    '''
    Encodes all given data, then classifies it into its sections
    '''
    for i in tqdm(range(X.shape[0]), mininterval=0.5, leave=None):
        classes[y[i]] += encode(X[i], features, values)
    return classes

def classify(X, features, values, vec_space):
    '''
    Classifies a set of points X (mxn matrix)
    '''
    y_hat = np.zeros(X.shape[0])
    for i in tqdm(range(X.shape[0]), mininterval=0.5, leave=None):
        sum = encode(X[i], features, values).quantise()
        y_hat[i] = vec_space.get(sum)
    return y_hat


class Classifier:

    vec_space = None
    features = None
    values = None

    P = None

    def __init__(self, P=None, type="BIN"):
        '''
        Classification initialisation method
        '''
        self.vec_space = VectorSpace(SIZE, type=type)
        if P is not None:
            self.P = P
        else:
            self.P = cpu_count() - 1

    def train(self, X, y):
        '''
        Start a multi-process classifier on X and y
        '''

        # Create HD vectors for every feature and value
        vecs = VectorGroups.randomVectors(X.shape[1])
        features = dict(zip(range(X.shape[1]), vecs))

        vecs = VectorGroups.levelVectors(X.max()-X.min()+1)
        values = dict(zip(range(X.min(), X.max()+1), vecs))

        # Create an empty HD point for every class
        unique_y = np.unique(y)
        vecs = VectorGroups.zeroVectors(len(unique_y))
        classes = dict(zip(unique_y, vecs))
        
        # Create task for sub-classifiers
        def task(X_sub, y_sub):
            f = deepcopy(features)
            v = deepcopy(values)
            c = deepcopy(classes)
            return train(X_sub, y_sub, f, v, c)

        # Start processes
        print(f"\n\nBEGIN TRAINING ON {self.P} PROCESSES")
        with Pool(self.P) as pool:
            cls = pool.map(task, np.array_split(X, self.P), 
                           np.array_split(y, self.P))

        # Merge trained vectors
        for i in classes.keys():
            for instance in cls:
                classes[i] += instance[i]
        
        # Insert these class vectors into the vector space
        for cl, vec in classes.items():
            self.vec_space.insertVector(vec, label=cl)

        # Save the features and values dicts to the main classifier
        self.features = features
        self.values = values

    def retrain(self, X, y):
        '''
        Retraining algorithm from LeHDC paper
        https://arxiv.org/pdf/2203.09680.pdf
        '''

        # Constants
        RATE = 10

        # Setup as integer vector space in case user mistakenly used binary
        self.vec_space.type = "INT"

        # Get the data for original training
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.5)
        
        # Start the original training
        self.train(X_train, y_train)

        # Start 20 part retraining
        X_sect = np.split(X_val, 20)
        y_sect = np.split(y_val, 20)
        for i in range(20):
            print(f"\n\nBEGIN RETRAINING {i+1}")
            X_data = X_sect[i]
            y_true = y_sect[i]
            y_pred = self.classify(X_data)
            for j in range(len(y_pred)):
                if y_pred[j] != y_true[j]:
                    encoded = encode(X_data[j], self.features, self.values)
                    true_ind = self.vec_space.v_labels.index(y_true[j])
                    pred_ind = self.vec_space.v_labels.index(y_pred[j])
                    self.vec_space.v_space[true_ind] += RATE * encoded
                    self.vec_space.v_space[pred_ind] -= RATE * encoded

    def classify(self, X):
        '''
        Classify a set of points X (mxn matrix) using multi-processing
        '''

        # Create task for sub-classifiers
        def task(X_sub):
            f = deepcopy(self.features)
            v = deepcopy(self.values)
            s = deepcopy(self.vec_space)
            return classify(X_sub, f, v, s)
        
        # Start processes
        print(f"\n\nBEGIN TESTING ON {self.P} PROCESSES")
        with Pool(self.P) as pool:
            cls = pool.map(task, np.array_split(X, self.P))
        
        # Merge output and return
        return np.concatenate(cls)

