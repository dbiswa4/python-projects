def TupleTest():
    '''
    A tuple is a sequence of immutable Python objects.
    The differences between tuples and lists are, the tuples cannot be 
    changed unlike lists and tuples use parentheses, whereas lists use square brackets.
    '''
    tup = ()
    tup1 = ('physics', 'chemistry', 1997, 2000)
    print "tup1 : ", tup1
    
    tup2 = (1, 2, 3, 4, 5 )
    print "tup2 : ", tup2
    tup3 = "a", "b", "c", "d"
    print "tup3 : ", tup3
    
    '''
    Like string indices, tuple indices start at 0, and they can be sliced, concatenated, and so on.
    '''
    print 'tup1[0] : ', tup1[0]
    
    # Following action is not valid for tuples
    # tup1[0] = 100;

    # So let's create a new tuple as follows
    tup3 = tup1 + tup2;
    print tup3


if __name__ == "__main__":
    print "Hello World"
    TupleTest()