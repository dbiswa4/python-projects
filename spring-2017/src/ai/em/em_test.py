import random
import numpy
import math
from copy import deepcopy

previous_estimated_cs = []
estimated_cs = numpy.random.uniform(0.0,1.0,5).tolist()

def convergeIsNotTrue():
    print ("previous_estimated_cs : ", previous_estimated_cs)
    print ("estimated_cs : ", estimated_cs)
    improvement = max([abs(estimated_cs - previous_estimated_cs)])
    print ("improvement:         ", improvement)
    if improvement > 0.05:
        return True
    else:
        return False


print convergeIsNotTrue()
