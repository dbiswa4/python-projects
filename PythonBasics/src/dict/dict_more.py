def dict_assignment():
    finaldict = {}
    for i in range(0,2):
        mydict = {}
        for j in range(0,2):
            mydict['key_' + str(j)] = 'first'
            mydict['key_' + str(j)] = 'second'
        finaldict[i] = mydict

    print 'Final dict: \n', finaldict

    '''
    {0: {'key_1': 'second', 'key_0': 'second'}, 1: {'key_1': 'second', 'key_0': 'second'}}
    '''

if __name__ == "__main__":
    #dictionary_test()
    dict_assignment()