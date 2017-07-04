d = {"a": 1, "b": 2}

#Both of the ways produces same result
for key in d.keys():
    print key

for key in d:
    print key
    print d[key]

#Another way
for key, val in d.items():
    print 'key   : ', key
    print 'value : ', val

#Calculate the sum of all dictionary values.
d = {"a": 1, "b": 2, "c": 3}

print 'sum : ', sum(d.values())
print 'd.values() return type : ', type(d.values())

#Filter the dictionary by removing all items with a value of greater than 1
d = {"a": 1, "b": 2, "c": 3}
d = dict((key, value) for key, value in d.items() if value <=1)
print 'Filtered dict: \n', d

#Create a dictionary of keys a, b, c where each key has as value a list from 1 to 10, 11 to 20, and 21 to 30
# respectively. Then print out the dictionary in a nice format.

from pprint import pprint
d = dict(a = list(range(1, 11)), b = list(range(11, 21)), c = list(range(21, 31)))
print d
pprint(d)