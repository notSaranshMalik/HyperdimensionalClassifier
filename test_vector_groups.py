from vector_groups import VectorGroups
import numpy as np
import matplotlib.pyplot as plt

# Global setup
V_SIZE = 10000
HD_group = VectorGroups()

# Definitions
def similarity_matrix(d):
    arr = np.empty((len(d), len(d)))
    for i in range(len(d)):
        for j in range(len(d)):
            arr[i, j] = np.sum(d[i]._vector == d[j]._vector)
    plt.matshow(arr)
    plt.show()

# Testing random and level vectors
random = HD_group.random_vectors(10)
similarity_matrix(random)

level = HD_group.level_vectors(10)
similarity_matrix(level)