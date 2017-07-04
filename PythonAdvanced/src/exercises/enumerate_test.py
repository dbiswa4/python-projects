'''
enumerate(a)  creates an enumerate object which yields pairs of indexes and items. Then we iterate through that object
print out the item-index pairs in each iteration together with some other strings
'''

a = [1, 2, 3]
for index, item in enumerate(a):
    print("Item %s has index %s" % (item, index))

