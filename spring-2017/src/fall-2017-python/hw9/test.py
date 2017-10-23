import numpy as np

def getAvg(scores):
    sum = np.sum(scores[1:])
    l = len(scores)
    avg = sum/float(l)
    print("len : ", l)
    print("sum : ", sum)
    print("avg : ", sum/float(l))
    return avg



nums = [1,2,3]
getAvg(nums)