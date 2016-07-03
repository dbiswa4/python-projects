def pass_example():
    '''
    It is used when a statement is required syntactically but you do not want any command or code to execute.
    The pass statement is a null operation; nothing happens when it executes. 
    The pass is also useful in places where your code will eventually go, but has not been written yet.
    '''
    for letter in 'Python': 
        if letter == 'h':
            pass
            print 'This is pass block'
        print 'Current Letter :', letter

    print "Good bye!"
    
if __name__ == "__main__":
    print "Hello World"
    pass_example()