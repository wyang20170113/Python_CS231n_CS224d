##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    eps = sqrt(6)/(sqrt(m + n))
    A0 = 2*eps*random.rand(m,n) - eps
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0