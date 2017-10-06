if __name__ == "__main__":
    print "Hello World"
    
    num = 30   
    
    div = num/(2 * 6)
    print div
    
    div = 556 / (1024 * 1024)
    
    print div

    num = 10

    print num/2
    print num%2

    num = 7

    print num/2
    print num%2

    listone = [1, 2, 3]
    listtwo = [4, 5, 6]
    mergedlist = []
    mergedlist.extend(listone)
    mergedlist.extend(listtwo)
    print mergedlist

    mergedlist = []
    mergedlist += listone
    mergedlist += listtwo
    print mergedlist


    mergedlist = []
    mergedlist.append(listone)
    mergedlist.append(listtwo)
    print mergedlist


