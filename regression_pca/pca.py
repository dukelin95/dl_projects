import numpy as np

from numpy import linalg as LA
from dataloader import Dataloader

class DimReduction():
    def __init__(self):
        pass

    def pca(self, data, p):
        '''
        Implements PCA (using Turk and Pentland trick) and returns the top p eigen values and vectors
        Assumes data is of shape (M x (hxw)), where M = number of images of height h and width w

        Returns:
        top_p_eig_values = size (p)
        top_p_eig_vectors = size (dxp), each column is a d dimensional eigen vector
        '''
        assert isinstance(data, np.ndarray)
        assert isinstance(p, int) and p > 0
        assert np.max(data) <= 1.0, 'pixel value range should be 0 to 1'

        M = data.shape[0] # M = number of images
        A = data.reshape(M, -1) # A = (Mxd)
        
        # Subtracing mean
        mean = np.mean(A, axis=0)
        A = A - mean #subtracing mean face from all data
        A = A.T # changing shape from (Mxd) to (dxM)

        # Eigen analysis
        eig_values, eig_vectors = LA.eig(A.T@A) #each column of eig_vectors is an eigen vector
        eig_vectors = A @ eig_vectors # TURK AND PENTLAND trick (dxM) x (MxM) = (dxM)
        
        sort_index = list(np.argsort(eig_values)) #sorting eigen values
        sort_index = sort_index[::-1] #descending order
        eig_values_sorted = eig_values[sort_index]
        eig_vectors_sorted = eig_vectors[:, sort_index] #sorting eigen vectors according to eigen values
        
        # Picking top p eigen values and vectors
        top_p_eig_values = eig_values_sorted[0:p]
        top_p_eig_vectors = eig_vectors[:, 0:p]
        
        return top_p_eig_values, top_p_eig_vectors
        

if __name__ == "__main__":
    dl = Dataloader("/Users/adnanshahpurwala/winter_2020/machine_leanring_review/regression/aligned/")
    tr, va, te = dl.get_k_fold(10, ['happiness', 'anger', 'disgust'])
    data1, target1 = tr[0]
    dim_reduction = DimReduction()
    eig_values, eig_vectors = dim_reduction.pca(data1, p=15)




