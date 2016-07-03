def sum( arg1, arg2 ):
    # Add both the parameters and return them."
    total = arg1 + arg2; # Here total is local variable.
    print "Inside the function local total : ", total
    return total, 3

Money = 2000
def AddMoney():
    # Uncomment the following line to fix the code:
    #UnboundLocalError: local variable 'Money' referenced before assignment
    #If below lone is commented, we get above error
    global Money
    Money = Money + 1
    
global_tuple = ()
def TupleTest():
    global global_tuple
    global_tuple = ("abc", "def")
    
def TupleReTest():
    global global_tuple
    print 'In TupleReTest()\n'
    print 'global_tuple : ', global_tuple
    

if __name__ == "__main__":
    print "Variable Scope...\n"
    
    total = 0; # This is global variable.
    tot, random = sum(10, 20)
    
    print "tot : ", tot
    print "random : ", random
    
    print "total : ", total
    
    #Another example
    print Money
    AddMoney()
    print Money
    
    print "Test Tuple"
    print "global_tuple : ", global_tuple
    TupleTest()
    print "global_tuple after calling TupleTest(): ", global_tuple
    
    TupleReTest()
    