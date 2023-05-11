import numpy as np
from vector_space import VectorSpace
from vector import Vector

SIZE = 10000

class Classifier:

    vec_space = None
    features = None
    values = None

    def __init__(self):
        '''
        Classification initialisation method
        '''
        self.vec_space = VectorSpace(SIZE)

    def train(self, X, y):
        '''
        Train the model on the data X (mxn matrix) and classification 
        y (mx1 matrix)
        '''

        # Logging setup
        print("LOG: Pre-processing")
        breaks = [int(X.shape[0]/10)*(i+1) for i in range(9)]

        # Create HD vectors for every feature and value
        features = dict()
        for i in range(X.shape[1]):
            features[i] = Vector(SIZE)

        values = dict()
        for i in np.unique(X):
            values[i] = Vector(SIZE)

        # Create a binding for every training point
        vecs = []
        for i in range(X.shape[0]):
            sum = Vector(SIZE, zero_vec=True)
            for j in range(X.shape[1]):
                sum += features[j] * values[X[i][j]]
            vecs.append(sum)
            if i in breaks: # logging
                print(f"LOG: {breaks.index(i)+1}0% encoding")
        print("LOG: 100% encoding") # logging

        # Create an empty HD point for every class
        classes = dict()
        for i in np.unique(y):
            classes[i] = Vector(SIZE, zero_vec=True)

        # Bundle all training vectors into the class vectors
        for i in range(y.size):
            classes[y[i]] += vecs[i]

        # Insert these class vectors into the vector space
        for cl, vec in classes.items():
            self.vec_space.insertVector(vec, label=cl)

        # Save the features and values dicts to the instance class to use
        # during classification
        self.features = features
        self.values = values

    def classify(self, X):
        '''
        Classify a set of points X (mxn matrix)
        '''

        # Logging setup
        breaks = [int(X.shape[0]/10)*(i+1) for i in range(9)]

        # Classification begins
        y_hat = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sum = Vector(SIZE, zero_vec=True)
            for j in range(X.shape[1]):
                sum += self.features[j] * self.values[X[i][j]]
            y_hat[i] = self.vec_space.get(sum)
            if i in breaks: # logging
                print(f"LOG: {breaks.index(i)+1}0% classified")
        print(f"LOG: 100% classified")
        
        return y_hat
