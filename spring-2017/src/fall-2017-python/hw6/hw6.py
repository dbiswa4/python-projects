import numpy as np
data = [-1, 3, 3, 4, 15, 16, 16, 17, 23, 24, 24, 25, 35, 36, 37, 46]

print(np.percentile(data, 25))
print(np.percentile(data, 50), np.median(data))
print(np.percentile(data, 75))

print 'sum : ', np.sum(data)
print 'size : ', np.size(data)