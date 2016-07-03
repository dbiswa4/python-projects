if __name__ == "__main__":
    print "Garbage Collection demo"
    
    #Note:
    #Good Link : Read "Destroying Objects (Garbage Collection)" 
    #www.tutorialspoint.com/python/python_classes_objects.htm
    #
    a = 40      # Create object <40>
    print "a : ", a
    b = a       # Increase ref. count  of <40> 
    print "b : ", b
    c = [b]     # Increase ref. count  of <40> 
    print "c : ", c

    del a       # Decrease ref. count  of <40>
    b = 100     # Decrease ref. count  of <40> 
    c[0] = -1   # Decrease ref. count  of <40> 