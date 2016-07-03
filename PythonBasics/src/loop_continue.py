def continue_example():
    print "\ncontinue_example()"
    for letter in 'Python':     # First Example
        if letter == 'h':
            continue
        print 'Current Letter :', letter

    var = 10                    # Second Example
    while var > 0:              
        var = var -1
        if var == 5:
            continue
        print 'Current variable value :', var
    print "Good bye!"
    
if __name__ == "__main__":
    print "Hello World"
    continue_example()