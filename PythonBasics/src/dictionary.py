import boto3
import operator
from collections import OrderedDict

def dictionary_test():
    #kind of hash table type
    #enclosed by curly braces ({ })
    #values can be assigned and accessed using square braces ([])
    #
    dict = {}
    dict['one'] = "This is one"
    dict[2]     = "This is two"

    tinydict = {'name': 'john','code':6734, 'dept': 'sales'}

    print dict['one']       # Prints value for 'one' key
    print dict[2]           # Prints value for 2 key
    print tinydict          # Prints complete dictionary
    print tinydict.keys()   # Prints all the keys
    print tinydict.values() # Prints all the values
    
def group_by_test():
    date = '2016-05-29'
    shard_size = 3456313439 
    shards_sizes = {}
    if shards_sizes.has_key(date):
        print "key found"
        shards_sizes[date] += shard_size
    else:
        print "key NOT found"
        shards_sizes[date] = shard_size
    
    date = '2016-05-30'
    shard_size = 4000000000
    
    if shards_sizes.has_key(date):
        print "key found"
        shards_sizes[date] += shard_size
    else:
        print "key NOT found"
        shards_sizes[date] = shard_size
    
    print shards_sizes.keys()
    print shards_sizes.values()
    
    #shards_sizes_sorted = sorted(shards_sizes.items(), key=operator.itemgetter(1), reverse=True)
    shards_sizes_sorted = sorted(shards_sizes.items(), key=operator.itemgetter(1))
    
    print "Sorted entries : ", shards_sizes_sorted
    
    #does not work
    #shards_sizes_sorted[0].values()
    
    #get the last one - works
    #print shards_sizes_sorted.pop()
    #print "after pop : ", shards_sizes_sorted
    shards_sizes_sorted = OrderedDict(sorted(shards_sizes.items(), key=lambda x: x[1], reverse=True))
    print "Again Sorted entries : ", shards_sizes_sorted
    print shards_sizes_sorted.keys()
    print shards_sizes_sorted.values()
    print shards_sizes_sorted.itervalues().next()
    
    #Sum all values
    print "Sum all the values : ", sum(shards_sizes_sorted.values())
    
    #Find max value
    print "key with max value : ", max(shards_sizes, key=shards_sizes.get)
    #Below one give weired result
    print "value with max value : ", max(shards_sizes, key=shards_sizes.get)[0] #2 !!!!
    #Below one works
    print "max value : ", shards_sizes[max(shards_sizes, key=shards_sizes.get)]
    
    print "key with max value : ", max(shards_sizes, key=lambda i: shards_sizes[i])
    
    avg_shard_size = sum(shards_sizes.values())/len(shards_sizes)
    print "Avg shard size : ", avg_shard_size
    
if __name__ == "__main__":
    #dictionary_test()
    group_by_test()