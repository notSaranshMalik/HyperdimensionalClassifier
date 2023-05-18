import numpy as np

class FeaturePicker:

    @staticmethod
    def calculateMatrix(X, y):

        X = (X - X.min()) / (X.max()-X.min())

        classes = dict()
        count = dict()
        for i in np.unique(y):
            classes[i] = np.zeros(X.shape[1])
            count[i] = 0

        for i in range(X.shape[0]):
            classes[y[i]] += X[i]
            count[y[i]] += 1

        for key, value in classes.items():
            classes[key] = value/count[key]

        return classes
    
    @staticmethod
    def pickFeatures(X, y):

        sim = FeaturePicker.calculateMatrix(X, y)

        importance = np.empty(X.shape[1])
        for feature in range(X.shape[1]):

            feat_avg = 0
            for cls in sim.values():
                feat_avg += cls[feature]
            feat_avg /= len(sim)

            feat_importance = False
            for cls in sim.values():
                if (cls[feature] - feat_avg) >= 0.05 or \
                      (cls[feature] - feat_avg) <= -0.05:
                    feat_importance = True
            importance[feature] = feat_importance
            
        return importance==1
