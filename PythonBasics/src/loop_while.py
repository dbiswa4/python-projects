def oneliner_while():
    print "\noneliner_while()"
    flag = 2
    while (flag == 1): print 'Given flag is really true!'
    else: print "Given flag is NOT really true!"
    print "Good bye!"

def while_again():
    print "\nwhile_again()..."
    count = 0
    while count < 5:    
        print count, " is  less than 5"
        count = count + 1
    else:
        print count, " is not less than 5"

def while_yet_again():
    print "\nwhile_yet_again()..."
    var = 1
    while var == 1 :  # This constructs an infinite loop
        num = raw_input("Enter a number  :")
        print "You entered: ", num

    print "Good bye!"

if __name__ == "__main__":
    print "Hello World"
    oneliner_while()
    while_again()
    while_yet_again()
    