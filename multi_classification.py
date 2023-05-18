import warnings
warnings.filterwarnings("ignore")

from classification import Classifier
from vector import Vector
import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from copy import deepcopy
from os import cpu_count

SIZE = 10000

class MultiprocessClassifier:

    main_classifier = None
    P = None

    def __init__(self, P=None):
        '''
        Classification initialisation method
        '''
        self.main_classifier = Classifier()
        if P is not None:
            self.P = P
        else:
            self.P = cpu_count() - 1

    def train(self, X, y, enc_zero=True):
        '''
        Start a multi-process classifier on X and y with enc_zero encoding
        '''

        # Create HD vectors for every feature and value
        features = dict()
        for i in range(X.shape[1]):
            features[i] = Vector(SIZE)

        values = dict()
        for i in np.unique(X):
            values[i] = Vector(SIZE)

        # Create an empty HD point for every class
        classes = dict()
        for i in np.unique(y):
            classes[i] = Vector(SIZE, zero_vec=True)
        
        # Create task for sub-classifiers
        def task(X_sub, y_sub, t_pos):
            cl = Classifier()
            f = deepcopy(features)
            v = deepcopy(values)
            c = deepcopy(classes)
            return cl._encode(X_sub, y_sub, f, v, c, enc_zero, t_pos)

        # Start processes
        print(f"\n\nBEGIN TRAINING ON {self.P} PROCESSES")
        with Pool(self.P) as pool:
            cls = pool.map(task, np.array_split(X, self.P), np.array_split(y, self.P), range(self.P))

        # Merge trained vectors
        for i in classes.keys():
            for instance in cls:
                classes[i] += instance[i]
        
        # Insert these class vectors into the vector space
        for cl, vec in classes.items():
            self.main_classifier.vec_space.insertVector(vec.quantise(), label=cl)

        # Save the features and values dicts to the main classifier
        self.main_classifier.features = features
        self.main_classifier.values = values
        self.main_classifier.enc_zero = enc_zero

    def classify(self, X):
        '''
        Classify a set of points X (mxn matrix) using multi-processing
        '''

        # Create task for sub-classifiers
        def task(X_sub, t_pos):
            cl = Classifier()
            f = deepcopy(self.main_classifier.features)
            v = deepcopy(self.main_classifier.values)
            s = deepcopy(self.main_classifier.vec_space)
            cl.features = f
            cl.values = v
            cl.vec_space = s
            cl.classes = self.main_classifier.enc_zero
            return cl.classify(X_sub, t_pos, no_print=True)
        
        # Start processes
        print(f"\n\nBEGIN TESTING ON {self.P} PROCESSES")
        with Pool(self.P) as pool:
            cls = pool.map(task, np.array_split(X, self.P), range(self.P))
        
        # Merge output and return
        return np.concatenate(cls)

