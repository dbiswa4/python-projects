def list_test():
    #all the items belonging to a list can be of different data type
    list = [ 'abcd', 786 , 2.23, 'john', 70.2 ]
    tinylist = [123, 'john']

    print list          # Prints complete list
    print list[0]       # Prints first element of the list
    print list[1:3]     # Prints elements starting from 2nd till 3rd; like java substring() - upper bound is excluded
    print list[2:]      # Prints elements starting from 3rd element
    print tinylist * 2  # Prints list two times
    print list + tinylist # Prints concatenated lists
    
if __name__ == "__main__":
    list_test()
    