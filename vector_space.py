import numpy as np
from vector import Vector

class VectorSpace:

    v_space = None
    v_labels = None
    v_size = None

    type = None

    def __init__(self, v_size, type="BIN"):
        '''
        Creates a hyperdimensional vector space of size v_size
        A "BIN" (binary) type vector space works by quantising and using 
        Hamming Distance
        An "INT" (integer) type vector space works by using Cosine Similarity
        '''
        self.v_space = []
        self.v_labels = []
        self.v_size = v_size
        self.type = type

    def newVector(self, label=None):
        '''
        Creates a new vector and adds it to the current vector space
        If the vector has a label, it is given that title
        '''
        vec = Vector(self.v_size)
        self.v_space.append(vec)
        self.v_labels.append(label)
        return vec
    
    def insertVector(self, vec, label=None):
        '''
        Adds a vector to the current vector space
        If the vector has a label, it is given that title
        '''
        if self.type == "BIN":
            self.v_space.append(vec.quantise())
        else:
            self.v_space.append(vec)
        self.v_labels.append(label)
    
    def get(self, vec):
        '''
        Probe for the vector in the vector space
        If the vector has a label, return that, else return the label
        '''
        size = len(self.v_space)
        dist = np.zeros(size)
        if self.type == "BIN":
            find = vec.quantise()
        else:
            find = vec
        for i in range(size):
            if self.type == "BIN":
                dist[i] = find.hammingDistance(self.v_space[i])
            elif self.type == "INT":
                dist[i] = -find.cosineSimilarity(self.v_space[i])
        if self.v_labels[dist.argmin()] is not None:
            return self.v_labels[dist.argmin()]
        return self.v_space[dist.argmin()]
    
    def newSequence(self, *v, label=None):
        '''
        Creates a new sequence using a multi permutation for each index 
        This is similar to ππππa + πππb + ππc + πd for a 4-sequence a, b, c, d
        If the sequence has a label, it is given that title
        '''
        perm = np.random.permutation(self.v_size)
        sum = Vector(self.v_size, zero_vec=True)
        for i in range(len(v)):
            sum += v[i].permute(perm, len(v)-i)
        sum = sum.quantise()
        self.v_space.append(sum)
        self.v_labels.append(label)
        return sum, perm
    