#Option-1
a = [1, "1", 1, "1", 2]
a = list(set(a))
print a #['1', 1, 2]

#The drawback here is that the original order of the items is lost.
# If you need to preserve the order you may want to use the solution in Answer 2 below.

#Option-2

from collections import OrderedDict
a = [1, "1", 1, "1", 2]
a = list(OrderedDict.fromkeys(a))
print a #[1, '1', 2]
print OrderedDict.fromkeys(a) #OrderedDict([(1, None), ('1', None), (2, None)])

#Option-2
a = [1, "1", 1, "1", 2]
b = []

for i in a:
    if i not in b:
        b.append(i)
#Order maintained
print b #[1, '1', 2]
