import numpy as np

class FeaturePicker:

    @staticmethod
    def calculateMatrix(X, y):
        '''
        Calculates a matrix of averages of where different classes show their
        points
        '''

        # Normalise data into a 0-1 range
        X = (X - X.min()) / (X.max()-X.min())

        # Setup individual matrices for the classes
        classes = dict()
        count = dict()
        for i in np.unique(y):
            classes[i] = np.zeros(X.shape[1])
            count[i] = 0

        # Update matrices with all data
        for i in range(X.shape[0]):
            classes[y[i]] += X[i]
            count[y[i]] += 1

        # Normalise and return
        for key, value in classes.items():
            classes[key] = value/count[key]
        return classes
    
    @staticmethod
    def pickFeatures(X, y):
        '''
        Picks the features that make a difference in classification to 
        eliminate noise
        '''

        # Pick features
        sim = FeaturePicker.calculateMatrix(X, y)

        # Find feature importance by checking if it makes at least a 5% 
        # difference
        importance = np.empty(X.shape[1])
        for feature in range(X.shape[1]):

            # Find avg usage of this feature
            feat_avg = 0
            for cls in sim.values():
                feat_avg += cls[feature]
            feat_avg /= len(sim)

            # Check if any class is more than 5% from the average
            feat_importance = False
            for cls in sim.values():
                if (cls[feature] - feat_avg) >= 0.05 or \
                      (cls[feature] - feat_avg) <= -0.05:
                    feat_importance = True
            importance[feature] = feat_importance
            
        return importance==1
