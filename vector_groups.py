import numpy as np
from vector import Vector

SIZE = 10000

class VectorGroups:

    @staticmethod
    def blank_vectors(n):
        '''
        Produces a set of uninitialised vectors of size n
        '''
        lst = []
        for _ in range(n):
            lst.append(Vector(SIZE, no_init=True))
        return lst

    @staticmethod
    def zero_vectors(n):
        '''
        Produces a set of zero vectors of size n
        '''
        lst = []
        for _ in range(n):
            lst.append(Vector(SIZE, zero_vec=True))
        return lst

    @staticmethod
    def random_vectors(n):
        '''
        Produces a set of random vectors of size n
        '''
        lst = []
        for _ in range(n):
            lst.append(Vector(SIZE))
        return lst
    
    @staticmethod
    def level_vectors(n):
        '''
        Produces a set of level vectors of size n
        '''

        prev = np.random.randint(2, size=SIZE)
        bit_flip = int(SIZE/n)

        lst = []
        for _ in range(n):
            flips = np.random.choice(SIZE, bit_flip, replace=False)
            prev[flips] = 1*(prev[flips] == 0)
            tmp = Vector(SIZE, no_init=True)
            tmp._vector = np.copy(prev)
            lst.append(tmp)
        return lst
    