import numpy as np
import math
# Preprocess
def toMiddle(A,B):
    '''
    put A to the middle of B
    '''
    Am,An,Bm,Bn = A.shape[0],A.shape[1],B.shape[0],B.shape[1]
    B[Bm//2-Am//2:Bm//2+math.ceil(Am/2),
      Bn//2-An//2:Bn//2+math.ceil(An/2)] = A
    return B
